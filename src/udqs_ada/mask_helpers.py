import h5py as h5
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import psutil
import os
import re

def calculate_chunk_size(dset, memory_fraction=0.001):
    if dset.ndim == 1:
        chunk_cols = 1
    else:
        chunk_cols = dset.shape[1]
    total_memory = psutil.virtual_memory().total
    available_memory = total_memory * memory_fraction
    element_size = 8
    elements_per_chunk = available_memory // element_size
    chunk_rows = int(elements_per_chunk // chunk_cols)
    print(f"Memory used pr chunk: {chunk_rows * chunk_cols * element_size / (1024**2):.2f} MB")
    if chunk_cols == 1:
        return (chunk_rows,)
    else:
        return (chunk_rows,chunk_cols)

def get_scan_step_mask(runnr, dataDir, memory_fraction=0.001, scan_step_array=None, n_scan_bins=None):
    if isinstance(runnr, (list, tuple, np.ndarray)):
        if isinstance(runnr,tuple):
            runnr = np.array(runnr)
        runnr.sort()
        run_str = "-".join(runnr)
        virtual_file_path = os.path.join(dataDir, f"Run-{run_str}_VirtualData.h5")
    else:
        virtual_file_path = os.path.join(dataDir, 'Run-' + runnr + '_VirtualData.h5')
    array_mask = None
    
    if scan_step_array is None:
        with h5.File(virtual_file_path, 'r+') as f:
            if "scan_step_mask" in f.keys():
                return f['scan_step_mask'][:]
                del f['scan_step_mask']
            dset = f['scan_step']
            chunk_shape = calculate_chunk_size(dset, memory_fraction=memory_fraction)
            dask_array = da.from_array(dset, chunks=chunk_shape)
            unique_values = da.unique(dask_array)
            unique_values = unique_values.compute()
            unique_values = unique_values[unique_values > -1]

            array_mask = da.zeros((unique_values.shape[0], dask_array.shape[0]), dtype=bool)
            for i in range(unique_values.shape[0]):
                array_mask[i] = (dask_array == unique_values[i])

            with ProgressBar():
                print('Computing the scan_step mask')
                array_mask = array_mask.compute()
            print('Writing computed array to file')
            f.create_dataset('scan_step_mask', shape=array_mask.shape, dtype=bool, data=array_mask)

        return array_mask
    else:
        # Compute a histogram-based scan_step_mask from the provided
        # 1D dataset inside the virtual file.
        with h5.File(virtual_file_path, 'r+') as f:
            if scan_step_array not in f:
                raise KeyError(f"scan_step_array '{scan_step_array}' not found in {virtual_file_path}")
            arr_dset = f[scan_step_array]
            if arr_dset.ndim != 1:
                raise ValueError(f"scan_step_array '{scan_step_array}' is not 1D")
            # Ensure same length as eventnr
            if 'eventnr' not in f:
                raise KeyError('eventnr dataset not found in virtual file')
            if arr_dset.shape[0] != f['eventnr'].shape[0]:
                raise ValueError(f"scan_step_array '{scan_step_array}' length {arr_dset.shape[0]} does not match 'eventnr' length {f['eventnr'].shape[0]}")

            # Load the array (1D). This is potentially large but needed to
            # compute bin assignment per event. If needed this could be
            # converted to a dask-based approach.
            arr = arr_dset[()]

            # Determine number of bins
            if n_scan_bins is None:
                # Prefer using existing scan_step_mask size if present
                if 'scan_step_mask' in f:
                    try:
                        existing_mask = f['scan_step_mask'][:]
                        n_bins = existing_mask.shape[0]
                        if n_bins < 1:
                            n_bins = None
                    except Exception:
                        n_bins = None
                else:
                    n_bins = None
                # If we couldn't get a sensible value, fallback to number
                # of unique values (capped) or sqrt heuristic.
                if n_bins is None:
                    uniq = np.unique(arr)
                    if uniq.shape[0] <= 1000:
                        n_bins = int(uniq.shape[0])
                    else:
                        n_bins = int(max(1, np.sqrt(arr.shape[0])))
            else:
                n_bins = int(n_scan_bins)

            # Compute bin edges and assign each event to a bin index
            edges = np.histogram_bin_edges(arr, bins=n_bins)
            # searchsorted yields indices in [0, len(edges)], subtract 1
            bin_idx = np.searchsorted(edges, arr, side='right') - 1
            # clip into range [0, n_bins-1]
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            # Build boolean mask: shape (n_bins, n_events)
            n_events = arr.shape[0]
            array_mask = np.zeros((n_bins, n_events), dtype=bool)
            for i in range(n_bins):
                array_mask[i] = (bin_idx == i)

            # Save/overwrite scan_step_mask in the virtual file so subsequent
            # calls can reuse it.
            if 'scan_step_mask' in f:
                try:
                    del f['scan_step_mask']
                except Exception:
                    pass
            f.create_dataset('scan_step_mask', shape=array_mask.shape, dtype=bool, data=array_mask)

            # Save/overwrite the scan_step bin edges as well for reference
            # Calculate centered bins
            edges = (edges[:-1] + edges[1:]) / 2.0
            # only scan_step_array last part for naming
            scan_step_array = scan_step_array.replace('/','_')
            if 'step_' + scan_step_array + '_bin_val' in f:
                try:
                    del f['step_' + scan_step_array + '_bin_val']
                except Exception:
                    pass
            f.create_dataset('step_' + scan_step_array + '_bin_val', shape=edges.shape, dtype=edges.dtype, data=edges)

        return array_mask
