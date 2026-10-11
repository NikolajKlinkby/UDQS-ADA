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

class ScanGroups:
    """Which events belong to which scan step, stored as two integer arrays.

    This replaces the boolean table of shape (n_steps, n_events): that table has exactly one
    True per event, so it holds no more than "the step of every event" while taking
    n_steps * n_events bytes.  Here the events are kept grouped by step:

        order   event numbers, grouped by step; ascending inside each step
        starts  where each step's group begins in `order` (length n_steps + 1)

    so the events of step i are order[starts[i]:starts[i+1]].  Memory is one integer per event
    plus one per step.
    """

    def __init__(self, order, starts, n_events):
        self.order = np.asarray(order, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.n_events = int(n_events)

    @property
    def n_steps(self):
        return len(self.starts) - 1

    @property
    def counts(self):
        """Number of events in each step."""
        return np.diff(self.starts)

    def events(self, i):
        """Event numbers of step i, in ascending order."""
        return self.order[self.starts[i]:self.starts[i + 1]]

    def filtered(self, use_mask):
        """Groups restricted by a user mask: per event (n_events,), or per step and event
        (n_steps, n_events) as the old boolean table allowed."""
        use_mask = np.asarray(use_mask)
        step_of = np.repeat(np.arange(self.n_steps), self.counts)      # step of every entry of `order`
        if use_mask.ndim == 1:
            if use_mask.shape[0] != self.n_events:
                raise ValueError(f"use_mask has {use_mask.shape[0]} entries but there are {self.n_events} events")
            keep = use_mask[self.order].astype(bool)
        elif use_mask.shape == (self.n_steps, self.n_events):
            keep = use_mask[step_of, self.order].astype(bool)
        else:
            raise ValueError(f"use_mask has shape {use_mask.shape}; expected ({self.n_events},) or ({self.n_steps}, {self.n_events})")
        kept_per_step = np.bincount(step_of[keep], minlength=self.n_steps)
        starts = np.concatenate(([0], np.cumsum(kept_per_step)))
        return ScanGroups(self.order[keep], starts, self.n_events)

    def to_mask(self):
        """The old boolean table (n_steps, n_events). Only for code that still wants it."""
        mask = np.zeros((self.n_steps, self.n_events), dtype=bool)
        mask[np.repeat(np.arange(self.n_steps), self.counts), self.order] = True
        return mask


def group_events(label, n_groups):
    """Group event numbers by an integer label per event (0 .. n_groups-1; negative = leave out)."""
    label = np.asarray(label)
    valid = np.nonzero(label >= 0)[0]
    by_step = np.argsort(label[valid], kind='stable')        # stable: events stay in ascending order
    order = valid[by_step]
    starts = np.searchsorted(label[order], np.arange(n_groups + 1))
    return ScanGroups(order, starts, label.shape[0])


def _virtual_path(runnr, dataDir):
    if isinstance(runnr, (list, tuple, np.ndarray)):
        if isinstance(runnr,tuple):
            runnr = np.array(runnr)
        runnr.sort()
        run_str = "-".join(runnr)
        return os.path.join(dataDir, f"Run-{run_str}_VirtualData.h5")
    return os.path.join(dataDir, 'Run-' + runnr + '_VirtualData.h5')


def get_scan_step_groups(runnr, dataDir, scan_step_array=None, n_scan_bins=None):
    """Events of every scan step, as a ScanGroups object.

    Without `scan_step_array` the steps are the distinct values of the `scan_step` dataset
    (values below 0 mark events outside the scan and are left out).  With `scan_step_array`
    the events are histogram-binned on that 1D dataset, one bin per step.

    Nothing large is stored: the grouping is one sort of a 1D array and is recomputed on
    every call, so it can never be out of date with the events in the file.
    """
    virtual_file_path = _virtual_path(runnr, dataDir)

    if scan_step_array is None:
        with h5.File(virtual_file_path, 'r+') as f:
            scan_step = f['scan_step'][:]
            unique_values = np.unique(scan_step)
            unique_values = unique_values[unique_values > -1]
            label = np.where(scan_step > -1, np.searchsorted(unique_values, scan_step), -1)
            groups = group_events(label, unique_values.shape[0])
            f.attrs['scan_step_rows'] = groups.n_steps
        return groups

    # Histogram-based steps from the provided 1D dataset inside the virtual file.
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

        arr = arr_dset[()]

        # Determine number of bins
        if n_scan_bins is None:
            # Prefer the number of steps of the previous grouping of this file, as before
            # (then recorded as the shape of the stored scan_step_mask, now as an attribute).
            n_bins = None
            if 'scan_step_rows' in f.attrs:
                n_bins = int(f.attrs['scan_step_rows'])
            elif 'scan_step_mask' in f:
                n_bins = int(f['scan_step_mask'].shape[0])
            if n_bins is not None and n_bins < 1:
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

        groups = group_events(bin_idx, n_bins)
        f.attrs['scan_step_rows'] = groups.n_steps

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

    return groups


def get_scan_step_mask(runnr, dataDir, memory_fraction=0.001, scan_step_array=None, n_scan_bins=None):
    """The boolean table (n_steps, n_events) that compute_scan_avg used to work with.

    Kept for code that calls it directly.  compute_scan_avg itself now uses
    get_scan_step_groups, which needs a tiny fraction of the memory.  The table is no longer
    written into the virtual file, and `memory_fraction` is not used any more.
    """
    return get_scan_step_groups(runnr, dataDir, scan_step_array=scan_step_array, n_scan_bins=n_scan_bins).to_mask()
