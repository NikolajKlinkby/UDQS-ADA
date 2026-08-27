import os
import gc
import re
import numpy as np
import h5py as h5
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, set_start_method
from tqdm import tqdm
import psutil
import dask.array as da
from dask.diagnostics import ProgressBar
import warnings

from .vds_helpers import (
    collect_from_meta,
    list_h5_datasets,
    _create_virtual_file,
)
from .mask_helpers import (get_scan_step_mask, calculate_chunk_size)
from .scan_helpers import (
    process_chunk,
    dry_run_process_chunk,
    find_dataset_containing,
    iterate_datasets,
    first_diff_index_simple,
)
from .model_helpers import process_model_chunk


def create_virtual_file(runnr, dataDir, fileprefix='File', overwrite=None):
    """
    Create (or combine) a virtual HDF5 dataset file for a run or multiple runs.

    If `runnr` is a single run identifier (string) this creates a single
    `Run-<runnr>_VirtualData.h5` file by scanning the files in
    `dataDir/Run-<runnr>`. If `runnr` is a list/tuple/ndarray of run ids the
    function ensures each run has a corresponding virtual file and then
    combines their virtual sources into a single combined virtual file
    `Run-<runnr1>-...-<runnrn>_VirtualData.h5`.

    Parameters:
    - runnr: str or sequence
        Single run identifier string or iterable of run identifier strings.
    - dataDir: str
        Directory containing `Run-<runnr>` subdirectories and HDF5 files.
    - fileprefix: str, optional
        Prefix of files to include (default 'File').
    - overwrite: bool or None, optional
        If True, overwrite existing virtual dataset files. If False, do not
        overwrite and only add missing sources. If None, prompt the user
        for interactive confirmation when necessary.

    Returns:
    - int
        Returns 1 when changes were written, 0 when no changes were needed.
    """
    # Support both single-run (string) and multi-run (sequence) inputs.
    # When a sequence is passed we ensure each run has a VDS and then
    # build a combined virtual file that concatenates their virtual sources.
    if isinstance(runnr, (list, tuple, np.ndarray)):
        if isinstance(runnr,tuple):
            runnr = np.array(runnr)
        runnr.sort()
        vds_files = []
        files_changed = 0
        for r in runnr:
            vds_files.append(os.path.join(dataDir, 'Run-' + r + '_VirtualData.h5'))
            files_changed += _create_virtual_file(r,dataDir, fileprefix=fileprefix, overwrite=overwrite)
        run_str = "-".join(runnr)
        combined_vds = os.path.join(dataDir, f"Run-{run_str}_VirtualData.h5")
        if files_changed == 0 and os.path.exists(combined_vds) and not overwrite:
            print('No change to virtual files')
            return 0
        vsource_lists = {}
        scan_val_dict = {}
        for vfile in vds_files:
            with h5.File(vfile, 'r') as f:
                meta_grp = f["vds_metadata"]
                vsource_meta_names = collect_from_meta(meta_grp)
                for dataset_name in vsource_meta_names:
                    item = meta_grp[dataset_name]
                    filenames = [fn.decode("utf-8") for fn in item["filenames"][:]]
                    dset_names = [dn.decode("utf-8") for dn in item["dset_names"][:]]
                    shapes     = [tuple(s) for s in item["shapes"][:]]
                    dtypes     = [np.dtype(dt.decode("utf-8")) for dt in item["dtypes"][:]]
                    # Recreate the VirtualSource entries recorded in the
                    # per-run VDS metadata so we can assemble a combined
                    # VirtualLayout below.
                    vsources = [
                        h5.VirtualSource(fn, dset, shape=shape, dtype=dt)
                        for fn, dset, shape, dt in zip(filenames, dset_names, shapes, dtypes)
                    ]
                    vsource_lists.setdefault(dataset_name, []).extend(vsources)
                for dset_name in list_h5_datasets(f):
                    if dset_name.strip('/') not in vsource_meta_names and dset_name.strip('/') != 'scan_step_mask':
                        dset = f[dset_name]
                        if isinstance(dset, h5.Dataset):
                            arr = dset[()]
                            scan_val_dict.setdefault(dset_name, []).append(arr)
        with h5.File(combined_vds, "w") as vds:
            meta_grp = vds.require_group("vds_metadata")
            for dataset_name, vsource_list in vsource_lists.items():
                layout_shape = (len(vsource_list),) + vsource_list[0].shape
                layout = h5.VirtualLayout(shape=layout_shape, dtype=vsource_list[0].dtype)
                for idx, vs in enumerate(vsource_list):
                    layout[idx] = vs
                vds.create_virtual_dataset(dataset_name, layout)
                # Record metadata describing the combined virtual sources
                # so the combined VDS can be introspected or reconstructed
                # later (e.g. filenames, dataset names, shapes, dtypes).
                filenames = np.array([vs.path.encode("utf-8") for vs in vsource_list])
                dset_names = np.array([vs.name.encode("utf-8") for vs in vsource_list])
                shapes = np.array([vs.shape for vs in vsource_list], dtype=np.int64)
                dtypes = np.array([str(vs.dtype).encode("utf-8") for vs in vsource_list])
                ds_grp = meta_grp.create_group(dataset_name)
                ds_grp.create_dataset("filenames", data=filenames)
                ds_grp.create_dataset("dset_names", data=dset_names)
                ds_grp.create_dataset("shapes", data=shapes)
                ds_grp.create_dataset("dtypes", data=dtypes)
            for dataset_name, arrays in scan_val_dict.items():
                vds.create_dataset(dataset_name, data=np.concatenate(arrays))
        print(f"Combined virtual dataset written: {combined_vds}")
        return 1
    else:
        return _create_virtual_file(runnr,dataDir, fileprefix=fileprefix, overwrite=None)

def compute_scan_avg(runnr, dataDir, arrays, n_std_filter=1.0, use_mask=None, memory_fraction=0.001, sum_mode=True, n_workers=4, 
                     scan_step_array=None, n_scan_bins=None, reference_arrays=None):
    """
    Compute scan-averaged datasets from a virtual dataset file and store the
    reduced scan results in a `Run-<runnr>_ReducedScan.h5` file.

    This function supports single or multiple run identifiers (which are
    combined into a single virtual file). It finds target datasets, applies a
    scan-step mask, and computes sums/medians or other aggregated quantities
    per scan-step using parallel worker processes. Temporary chunk files are
    created under `tmp_chunks_<run>` during processing and cleaned up on
    completion.

    Parameters:
    - runnr: str or sequence
        Run identifier or list/array of identifiers to combine.
    - dataDir: str
        Directory containing virtual and raw run files.
    - arrays: str or sequence
        Dataset path(s) inside the virtual HDF5 file to process.
    - n_std_filter: float, optional
        Number of standard deviations used to define an acceptance window
        around the median when filtering outliers (default 1.0).
    - use_mask: ndarray or None, optional
        Boolean mask to combine with computed scan-step mask.
    - memory_fraction: float, optional
        Fraction used to request memory for mask calculation.
    - sum_mode: bool, optional
        If True compute aggregated sums/Counts; otherwise do per-step storage.
    - n_workers: int, optional
        Maximum number of parallel worker processes to spawn.
    - scan_step_array: str or None, optional
        If provided, use this 1D dataset path inside the virtual file to
        compute scan steps via histogram binning instead of the existing
        `scan_step` dataset.
    - n_scan_bins: int or None, optional
        Number of bins to use when computing scan steps from `scan_step_array`.
    - reference_arrays: [str,str] or sequence of [str,str] or None, optional
        If provided, use these array pairs to compute per-scan-step reference

    Returns:
    - None
        Results are written into `Run-<runnr>_ReducedScan.h5` inside `dataDir`.
    """
    # High-level steps:
    # 1. Expand input array/group paths to concrete dataset paths.
    # 2. Copy 1D scalar datasets straight into the reduced file.
    # 3. Compute the scan-step mask (can be combined with `use_mask`).
    # 4. Spawn parallel workers to process spatial chunks and write
    #    temporary HDF5 chunk files.
    # 5. Merge the chunk outputs back into the final reduced HDF5.
    
    # Determine file paths and prepare temporary directory
    if isinstance(runnr, (list, tuple, np.ndarray)):
        if isinstance(runnr,tuple):
            runnr = np.array(runnr)
        runnr.sort()
        run_str = "-".join(runnr)
        virtual_file_path = os.path.join(dataDir, f"Run-{run_str}_VirtualData.h5")
        scan_file_path = os.path.join(dataDir, f'Run-{run_str}_ReducedScan.h5')
        tmp_dir = os.path.join(dataDir, f"tmp_chunks_{run_str}")
    # Single run case
    else:
        virtual_file_path = os.path.join(dataDir, 'Run-' + runnr + '_VirtualData.h5')
        scan_file_path = os.path.join(dataDir, f'Run-{runnr}_ReducedScan.h5')
        tmp_dir = os.path.join(dataDir, f"tmp_chunks_{runnr}")
    
    # Step 1: Expand input array/group paths to concrete dataset paths.
    if not isinstance(arrays,(tuple,list,np.ndarray)):
        arrays = [arrays]
    with h5.File(virtual_file_path, "r") as f:
        expanded_arrays = []
        for arr in arrays:
            # Determine whether the given path is a Group or Dataset
            # inside the virtual file so we can expand groups into
            # a list of datasets to process.
            obj_type = type(f.get(arr, getclass=True))
            if arr in f and isinstance(obj_type, h5.Group) and 'vds_metadata' not in arr:
                expanded_arrays.extend(list_h5_datasets(f, arr))
            elif arr == "/" or arr == "":
                expanded_arrays.extend(list_h5_datasets(f, "/"))
            elif arr in f and isinstance(obj_type, h5.Dataset) and 'vds_metadata' not in arr:
                expanded_arrays.append(arr)
            else:
                raise ValueError(f"Invalid array/group path: {arr}")
    gc.collect()
    arrays = expanded_arrays

    print(f"Found {len(arrays)} datasets to process.")
    
    # Step 2: Copy 1D scalar datasets straight into the reduced file.
    # By default create `scan_step_mask` using the existing `scan_step`
    # dataset. If `scan_step_array` is provided it will be used to build
    # the scan steps by histogram-binning that 1D array (one bin == one
    # scan step). Optionally provide `n_scan_bins` to override bin count.
    mask = get_scan_step_mask(runnr, dataDir, memory_fraction=memory_fraction, scan_step_array=scan_step_array, n_scan_bins=n_scan_bins)
    
    # Check validity of computed mask
    if isinstance(use_mask, np.ndarray):
        if use_mask.sum() == 0:
            raise ValueError("Mask has no truth values")
        mask = mask * use_mask
    # `mask` is a boolean array (per-scan-step) used to include/exclude
    # events during aggregation. If `use_mask` is provided it is applied
    # elementwise (logical AND) with the computed mask.

    # Copy/appended 1D datasets directly
    with h5.File(virtual_file_path, "r") as f_in, h5.File(scan_file_path, "a") as f_out:
        one_d_arrays = []
        for arr_name in arrays:
            data = f_in[arr_name]
            if data.ndim == 1:
                one_d_arrays.append(arr_name)
                print(f'Processing array: {arr_name}')
                existing_len = f_out[arr_name].shape[0] if arr_name in f_out else 0
                new_data_len = data.shape[0] - existing_len
                if new_data_len <= 0:
                    continue
                data_array = data[existing_len:]
                if data.dtype == 'O':
                    dtype = h5.string_dtype(encoding='ascii')
                else:
                    dtype = data.dtype
                if arr_name in f_out:
                    existing_data = f_out[arr_name][:]
                    data_array = np.concatenate([existing_data, data_array])
                if arr_name in f_out:
                    del f_out[arr_name]
                f_out.create_dataset(arr_name, shape=data_array.shape, dtype=dtype, data=data_array)
        arrays = [arr for arr in arrays if arr not in one_d_arrays]
    
    # 1D datasets are handled above (copied/appended) — remove them from
    # the list of arrays requiring spatial aggregation.
    if not arrays:
        print("All datasets were 1D — processing complete.")
        return
    
    array_pairs = []
    if reference_arrays is not None:
        if not isinstance(reference_arrays[0], (list, tuple)):
            reference_arrays = [reference_arrays]
        
        for ref_pair in reference_arrays:
            # If two arrays in arrays contain the reference pair 
            # and are identical when the reference pair is removed from the string
            ref_nom, ref_denom = ref_pair
            arr_nom = [arr for arr in arrays if ref_nom in arr]
            arr_denom = [arr for arr in arrays if ref_denom in arr]
            for a in arr_nom:
                prefix_nom = a.replace(ref_nom, '')
                for b in arr_denom:
                    prefix_denom = b.replace(ref_denom, '')
                    if prefix_nom == prefix_denom:
                        # The pairs match — store them as a pair to be processed together
                        array_pairs.append([a,b])
                        # Remove them from the main arrays list to avoid double-processing
                        arrays.remove(a)
                        arrays.remove(b)
    # Add the pairs to the arrays list for processing
    for pair in array_pairs:
        arrays.append(pair)

    # Make an array of array names
    array_names = []
    for arr in arrays:
        if isinstance(arr, (list, tuple)):
            array_names.append(arr[0] + '-' + arr[1][first_diff_index_simple(arr[0], arr[1]):])
        else:
            array_names.append(arr)

    # Step 3: Compute the scan-step mask (can be combined with `use_mask`).
    remaining_events = []
    if sum_mode:
        with h5.File(scan_file_path, 'a') as f_out:
            for arr_name in array_names:
                matches = find_dataset_containing(f_out, arr_name)
                if matches:
                    if 'Events' in f_out[matches[0]].attrs.keys():
                        remaining_events.append(f_out[matches[0]].attrs['Events'])
                    else:
                        remaining_events.append(0)
                else:
                    remaining_events.append(0)
    else:
        remaining_events = np.zeros(len(arrays))
    
    # Remove any existing temporary chunk files
    if os.path.exists(tmp_dir):
        for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    os.makedirs(tmp_dir, exist_ok=True)

    # Measure per-worker memory usage via a dry-run
    set_start_method("spawn", force=True)
    print('Measuring worker size')
    with Manager() as manager:
        max_mem = 0
        
        # Run a single dry-run worker to estimate per-worker memory needs.
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(
                    dry_run_process_chunk, 0, [0,0,0,0], mask, arrays, remaining_events, virtual_file_path, tmp_dir,
                    sum_mode, n_std_filter
                )
            ]
            for future in as_completed(futures):
                mem = future.result()
                if mem > max_mem:
                    max_mem = mem
    max_mem *= 2
    print(f'Assuming worker size {max_mem:.2f} MB')
    # Run a single dry-run worker to estimate per-worker memory needs.
    # This allows computing a safe number of parallel workers given
    # current available system memory.

    # Determine safe number of workers based on available memory
    available_mem = psutil.virtual_memory().available / (1024 ** 2) * 0.8
    safe_n_workers = max(1, int(available_mem // max_mem))
    if n_workers > safe_n_workers:
        print(f'Can not allocate {n_workers} workers. Using maximum workers {safe_n_workers}')
        n_workers = safe_n_workers
    indices_chunks = np.array_split(np.arange(mask.shape[0]), n_workers)
    
    # Step 4: Spawn parallel workers to process spatial chunks and write
    with Manager() as manager:
        # Setup shared counters and locks for progress tracking
        dataset_counters = {arr: manager.Value('i', 0) for arr in array_names}
        dataset_locks = {arr: manager.Lock() for arr in array_names}
        dataset_totals = {arr: mask.shape[0] for arr in array_names}
        tmp_paths = []
        tmp_indices = []
        arr_names = []

        # Spawn parallel workers to process spatial chunks and write
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for arr_name, arr, remaining_idx in zip(array_names, arrays, remaining_events):
                for idx, chunk in enumerate(indices_chunks):
                    futures.append(
                            executor.submit(
                                process_chunk, idx, chunk, mask, arr, remaining_idx,
                                virtual_file_path, tmp_dir, sum_mode, n_std_filter,
                                dataset_counters[arr_name], dataset_locks[arr_name]
                            )
                    )
            active_bars = {}

            # Monitor progress and update progress bars
            while any(not f.done() for f in futures):
                for arr in array_names:
                    with dataset_locks[arr]:
                        val = dataset_counters[arr].value
                    total = dataset_totals[arr]
                    if val >= total and arr in active_bars:
                        active_bars[arr].update(total-active_bars[arr].n)
                        active_bars[arr].close()
                        del active_bars[arr]
                        continue
                    if 0 < val < total:
                        if arr not in active_bars:
                            active_bars[arr] = tqdm(total=total, desc=f"Processing {arr}", unit='Steps')
                        bar = active_bars[arr]
                        delta = val - bar.n
                        if delta > 0:
                            bar.update(delta)

            # Close any progress bars that remain open after workers finish.
            for bar in active_bars.values():
                bar.close()

            # Collect results from completed futures
            for future in as_completed(futures):
                fut_tmp_paths, fut_indicies, arr_name = future.result()
                tmp_paths.append(fut_tmp_paths)
                tmp_indices.append(fut_indicies)
                arr_names.append(arr_name)
    
    # Step 5: Merge the chunk outputs back into the final reduced HDF5.
    with h5.File(scan_file_path, 'a') as f_out:
        for tmp_path, chunk_indices, arr_name in zip(tmp_paths, tmp_indices, arr_names):
            with h5.File(tmp_path, 'r') as f_tmp:
                # Check if the chunk processing was completed successfully
                if not f_tmp.attrs.get('done', False):
                    print(f"[Merge] Skipping incomplete chunk {tmp_path}")
                    continue
                
                # Merge datasets from the temporary chunk file into the final output
                for key, dset_tmp in iterate_datasets(f_tmp):
                    
                    # Create or resize the output dataset as needed
                    if key not in f_out:
                        chunks = (min(1000,mask.shape[0]),) + dset_tmp.shape[1:]
                        f_out.create_dataset(
                            key,
                            shape=(mask.shape[0],) + dset_tmp.shape[1:],
                            dtype=dset_tmp.dtype,
                            chunks=chunks,
                            maxshape=(None,) + dset_tmp.shape[1:],
                        )
                    else:
                        if f_out[key].shape[0] < mask.shape[0]:
                            f_out[key].resize((mask.shape[0],) + dset_tmp.shape[1:])
                    
                    # Merge data from the temporary dataset into the output dataset
                    dset_out = f_out[key]
                    dset_out.attrs.modify('Events', mask.shape[1])
                    for local_idx, global_idx in zip(range(dset_tmp.shape[0]), chunk_indices):
                        # Handle sum/Count datasets by accumulating values
                        if key.endswith('_count') or key.endswith('_sum'):
                            if dset_out.ndim == 1:
                                dset_out[global_idx] += dset_tmp[local_idx]
                            else:
                                dset_out[global_idx, ...] += dset_tmp[local_idx, ...]
                        # Handle mean/median datasets by direct assignment
                        else:
                            if dset_out.ndim == 1:
                                dset_out[global_idx] = dset_tmp[local_idx]
                            else:
                                dset_out[global_idx, ...] = dset_tmp[local_idx, ...]
            print(f"[Merge] Finished merging {tmp_path}")
            os.remove(tmp_path)
    
    # remove the contents of the temporary directory
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(tmp_dir)
    gc.collect()

def compute_model(runnr, dataDir, arrays, n_workers=6, method='projection', 
                harmonics=(0,4,8), reg=1e-8, robust=False, 
                use_scan_avg=False, angle_index = 1):
    """
    Compute model parameters from angular-resolved datasets using parallel
    processing and write results to `Run-<runnr>_ModelData.h5`.

    This function reads datasets from the virtual dataset file and computes
    model coefficients (for example via projection) for each spatial index.
    It splits the work into chunks, launches worker processes that run
    `process_model_chunk`, and merges chunk outputs into a final model HDF5
    file. Temporary chunk files are written under `tmp_chunks_<run>`.

    Parameters:
    - runnr: str or sequence
        Single run identifier or iterable of runs to combine.
    - dataDir: str
        Directory containing the run virtual file and where model output is
        written.
    - arrays: sequence
        List of dataset paths (within virtual file) to model.
    - n_workers: int, optional
        Number of parallel worker processes.
    - method: str, optional
        Modeling method name passed to `process_model_chunk` (e.g. 'projection').
    - harmonics: tuple, optional
        Harmonics to include in model fitting.
    - reg: float, optional
        Regularization parameter for model inversion.
    - weights: array-like or None, optional
        Per-angle weights passed to the chunk processing routine.
    - robust: bool, optional
        Whether to use robust fitting options.
    - use_scan_avg: bool, optional
        If True, use reduced-scan data during model calculations.
    - angle_index: int, optional
        Index of the angle dimension in the datasets (default 1).

    Returns:
    - None
        Model datasets are written into `Run-<runnr>_ModelData.h5`.
    """
    # NOTE: The `robust` option is currently not implemented.
    # If `robust=True` is passed, a UserWarning will be issued and
    # the function will proceed with `robust=False` behavior.
    # This is intentionally non-fatal to preserve existing workflows.
    if robust:
        warnings.warn(
            "compute_model: parameter `robust=True` is implemented wrong and causes crashes; proceeding with robust=False",
            UserWarning,
        )
        robust = False
    # Determine file paths and prepare temporary directory
    if isinstance(runnr, (list, tuple, np.ndarray)):
        if isinstance(runnr,tuple):
            runnr = np.array(runnr)
        runnr.sort()
        run_str = "-".join(runnr)
        run_prefix = f"Run-{run_str}"
        tmp_dir = os.path.join(dataDir, f"tmp_chunks_{run_str}")
    else:
        run_prefix = f"Run-{runnr}"
        tmp_dir = os.path.join(dataDir, f"tmp_chunks_{runnr}")

    # Define file paths
    virtual_file_path = os.path.join(dataDir,run_prefix + '_VirtualData.h5')
    model_file_path = os.path.join(dataDir,run_prefix+"_ModelData.h5")
    reduced_scan_path = os.path.join(dataDir,run_prefix+"_ReducedScan.h5")

    # Step 1: Expand input array/group paths to concrete dataset paths.
    if not isinstance(arrays,(tuple,list,np.ndarray)):
        arrays = [arrays]
    
    array_path = virtual_file_path if not use_scan_avg else reduced_scan_path
    with h5.File(array_path, "r") as f:
        expanded_arrays = []
        for arr in arrays:
            # Determine whether the given path is a Group or Dataset
            # inside the virtual file so we can expand groups into
            # a list of datasets to process.
            obj_type = f.get(arr, getclass=True)
            if arr in f and issubclass(obj_type, h5.Group) and 'vds_metadata' not in arr:
                datasets_list = list_h5_datasets(f, arr)
                # Check that the datasets have a name matching the expected '_mean' or '_sum' suffix
                # only include those datasets if they exist. otherwise use all datasets in the group
                valid_datasets = []
                for ds in datasets_list:
                    if re.match(r'.*(_mean|_sum)$', ds):
                        valid_datasets.append(ds)
                if len(valid_datasets) == 0:
                    valid_datasets = datasets_list
                expanded_arrays.extend(valid_datasets)
            elif arr == "/" or arr == "":
                datasets_list = list_h5_datasets(f, "/")
                valid_datasets = []
                for ds in datasets_list:
                    if re.match(r'.*(_mean|_sum)$', ds):
                        valid_datasets.append(ds)
                if len(valid_datasets) == 0:
                    valid_datasets = datasets_list
                expanded_arrays.extend(valid_datasets)
            elif arr in f and issubclass(obj_type, h5.Dataset) and 'vds_metadata' not in arr:
                expanded_arrays.append(arr)
            else:
                raise ValueError(f"Invalid array/group path: {arr}")
    gc.collect()
    # Use only datasets with expected suffixes if found otherwise try original list
    if len(expanded_arrays) != 0:
        arrays = expanded_arrays
    
    # get the shape of the datasets and make sure the model_file is created
    with h5.File(array_path, 'r') as f_scan, h5.File(model_file_path, 'w') as f_m:
        # try candidate names: arr_name, arr_name + '_mean', arr_name + '_sum'
        candidates = [arrays[0], f"{arrays[0]}_mean", f"{arrays[0]}_sum"]
        dset = None
        for c in candidates:
            if c in f_scan:
                dset = f_scan[c]
                break
        if dset is None or not isinstance(dset, h5.Dataset):
            raise KeyError(f"Could not find reduced dataset for '{arrays[0]}' in {array_path}; tried: {candidates}")
        
        # Get shape from the first array
        shape = dset.shape
        del dset
    # Make some static arrays used in the modeling
    angles = np.linspace(0, np.pi * 2, shape[angle_index])

    # Check all arrays have the same leading dimension
    arrays_to_remove = []
    with h5.File(array_path, 'r') as f_scan:
        for arr_name in arrays:
            dset = f_scan[arr_name]
            if dset.shape[angle_index] != shape[angle_index]:
                # Remove array from list if leading dimension does not match
                arrays_to_remove.append(arr_name)
    
    for arr_name in arrays_to_remove:
        arrays.remove(arr_name)
    
    # Prepare temporary directory
    if os.path.exists(tmp_dir):
        for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare and launch worker processes
    indices_chunks = np.array_split(np.arange(shape[0]), n_workers)
    set_start_method("spawn", force=True)

    # Setup manager to track progress across workers
    with Manager() as manager:
        dataset_counters = {arr: manager.Value('i', 0) for arr in arrays}
        dataset_locks = {arr: manager.Lock() for arr in arrays}
        dataset_totals = {arr: shape[0] for arr in arrays}
        tmp_paths = []
        tmp_indices = []
        arr_names = []

        # Launch worker processes to compute model chunks
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for arr_name in arrays:
                    for idx, chunk in enumerate(indices_chunks):
                        futures.append(
                            executor.submit(
                                process_model_chunk,
                                idx,
                                chunk,
                                arr_name,
                                angles,
                                virtual_file_path,
                                tmp_dir,
                                dataset_counters[arr_name],
                                dataset_locks[arr_name],
                                method=method,
                                harmonics=harmonics,
                                reg=reg,
                                weights=None,
                                robust=robust,
                                use_scan_avg=use_scan_avg,
                                reduced_scan_path=reduced_scan_path,
                                angle_index=angle_index,
                            )
                        )
            active_bars = {}

            # Monitor progress of workers and update progress bars
            while any(not f.done() for f in futures):
                for arr in arrays:
                    with dataset_locks[arr]:
                        val = dataset_counters[arr].value
                    total = dataset_totals[arr]
                    if val >= total and arr in active_bars:
                        active_bars[arr].update(total-active_bars[arr].n)
                        active_bars[arr].close()
                        del active_bars[arr]
                        continue
                    if 0 < val < total:
                        if arr not in active_bars:
                            active_bars[arr] = tqdm(total=total, desc=f"Processing {arr}", unit='Steps')
                        bar = active_bars[arr]
                        delta = val - bar.n
                        if delta > 0:
                            bar.update(delta)

            # Close any remaining progress bars after completion
            for bar in active_bars.values():
                bar.close()

            # Collect results from completed futures
            for future in as_completed(futures):
                fut_tmp_paths, fut_indicies, arr_name = future.result()
                tmp_paths.append(fut_tmp_paths)
                tmp_indices.append(fut_indicies)
                arr_names.append(arr_name)

    with h5.File(model_file_path, 'a') as f_out:
        for tmp_path, chunk_indices, arr_name in zip(tmp_paths, tmp_indices, arr_names):
            with h5.File(tmp_path, 'r') as f_tmp:
                for key, dset_tmp in iterate_datasets(f_tmp):
                    if key not in f_out:
                        chunks = (min(1000,shape[0]),) + dset_tmp.shape[1:]
                        f_out.create_dataset(
                            key,
                            shape=(shape[0],) + dset_tmp.shape[1:],
                            dtype=dset_tmp.dtype,
                            chunks=chunks,
                            maxshape=(None,) + dset_tmp.shape[1:],
                        )
                    else:
                        if f_out[key].shape[0] < shape[0]:
                            f_out[key].resize((shape[0],) + dset_tmp.shape[1:])
                    dset_out = f_out[key]
                    for local_idx, global_idx in zip(range(dset_tmp.shape[0]), chunk_indices):
                        dset_out[global_idx, ...] = dset_tmp[local_idx, ...]
            print(f"[Merge] Finished merging {tmp_path}")
            os.remove(tmp_path)
    
    # if tmp directory is empty then remove it
    if os.path.exists(tmp_dir):
        if len(os.listdir(tmp_dir)) == 0:
            os.rmdir(tmp_dir)
    gc.collect()

    # Move the step values from the virtual file to the model file
    with h5.File(virtual_file_path, 'r') as f_vds, h5.File(model_file_path, 'a') as f_model:
        # Find all datasets in the top level of the virtual file starting with 'step_'
        for dset_name in list_h5_datasets(f_vds):
            if dset_name.startswith('/step_'):
                data = f_vds[dset_name][:]
                if dset_name in f_model:
                    del f_model[dset_name]
                f_model.create_dataset(dset_name, data=data)

def mask_data(runnr, dataDir, mask, memory_fraction=0.001):
    if isinstance(runnr, (list, tuple, np.ndarray)):
        if isinstance(runnr,tuple):
            runnr = np.array(runnr)
        runnr.sort()
        run_str = "-".join(runnr)
        virtual_file_path = os.path.join(dataDir, f"Run-{run_str}_VirtualData.h5")
    else:
        virtual_file_path = os.path.join(dataDir, 'Run-' + runnr + '_VirtualData.h5')
    array_mask = None
    with h5.File(virtual_file_path, 'r') as f:
        dset = f['eventnr']
        chunk_shape = calculate_chunk_size(dset, memory_fraction=memory_fraction)
        dask_array = da.from_array(dset, chunks=chunk_shape)

        array_mask = da.ones(dask_array.shape[0], dtype='bool')
        for mask_key in mask.keys():
            if mask_key in f:
                mask_string = mask[mask_key]
                array = da.from_array(f[mask_key])
                evaluated_mask = eval(mask_string)
                if evaluated_mask.shape != array.shape:
                    raise ValueError(f"Mask shape {evaluated_mask.shape} does not match dataset shape {array.shape}")
                if evaluated_mask.dtype != 'bool':
                    raise ValueError(f"Mask for {mask_key} is not a boolean array")
                if not isinstance(evaluated_mask, da.Array):
                    raise ValueError(f"Mask for {mask_key} is not a dask array")
                array_mask *= evaluated_mask
            elif mask_key.lower() == 'condition':
                mask_string = mask[mask_key]
                m = re.match(r"(.+?)([<>=!]+)(.+)", mask_string.strip())
                if not m:
                    raise ValueError(f"Could not parse condition: {mask_string.strip()}")

                left_key, operator, right_key = [s.strip() for s in m.groups()]

                left_array = da.from_array(f[left_key], chunks=chunk_shape)
                right_array = da.from_array(f[right_key], chunks=chunk_shape)

                ops = {
                    "<": left_array < right_array,
                    "<=": left_array <= right_array,
                    ">": left_array > right_array,
                    ">=": left_array >= right_array,
                    "==": left_array == right_array,
                    "!=": left_array != right_array,
                }

                if operator not in ops:
                    raise ValueError(f"Unsupported operator {operator} in {mask_string.strip()}")

                evaluated_mask = ops[operator]

                if evaluated_mask.shape[0] == array_mask.shape[0]*2:
                    evaluated_mask = evaluated_mask[::2]

                array_mask &= evaluated_mask
            else:
                print(f"Group {mask_key} does not exist")

        with ProgressBar():
            print('Computing mask')
            array_mask = array_mask.compute()

    return array_mask