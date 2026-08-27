Data Analysis API
=================

Overview
--------
The `UDQS-ADA` package provides utilities to build and process
virtual HDF5 datasets for angular- and spatially-resolved pump-probe
measurements. It orchestrates creation of Virtual Dataset (VDS) files,
computation of scan-averaged reduced datasets, model parameter extraction
from angular-resolved data, and creation of boolean masks for event
selection.

Installation
------------
Install the package and its dependencies using pip from the pyproject.toml:
```bash
pip install -e .
```

Main functions
--------------
- `create_virtual_file(runnr, dataDir, fileprefix='File', overwrite=None)`:
	Create per-run Virtual HDF5 files (VDS) from physical HDF5 files in
	`dataDir/Run-<runnr>/` and optionally combine multiple runs into a
	concatenated VDS. Returns 1 when changes were written, 0 otherwise.

- `compute_scan_avg(runnr, dataDir, arrays, ...)`:
	Compute scan-averaged and aggregated datasets from a VDS and write a
	`Run-<runnr>_ReducedScan.h5` file. Supports outlier filtering, masking,
	chunked/parallel processing, and optional reference-array pairs.

- `compute_model(runnr, dataDir, arrays, ...)`:
	Compute model parameters (e.g. harmonic projection fits) from angular
	datasets and write a `Run-<runnr>_ModelData.h5` file. Uses worker
	processes and merges chunk outputs into a final HDF5 file.

- `mask_data(runnr, dataDir, mask, memory_fraction=0.001)`:
	Evaluate complex logical masks defined over dataset columns using Dask
	and return a boolean array used to include/exclude events during
	aggregation.

Package layout
--------------
- `vds_helpers.py`       — helpers for creating and inspecting virtual datasets
- `mask_helpers.py`      — mask computation and memory/chunk heuristics
- `scan_helpers.py`      — chunk processing and scan-averaging logic
- `model_helpers.py`     — per-chunk model fitting routines
- `api.py`               — high-level orchestrator functions (this package's API)

Quick usage examples (from included scripts)
------------------------------------------

Example: single run with scan averaging and model

```python
import udqs_ada as api

data_folder = "data_folder"
runnr = ['068']

# Create virtual dataset for the run (does not overwrite existing VDS)
api.create_virtual_file(runnr, data_folder, overwrite=False)

# Compute reduced scan averages (per-scan step), use the mask and reference arrays
#
# arrays=['/'] or '/' is the root path in the h5 file and will compute scan 
# averages for all datasets in all branching groups. 
# Otherwise point to a specific group to compute all datasets in the group or
# point to specific datasets.
#
# sum_mode is default True and simply sums all the values together and create a 
# '_count' array if one wants the average. This is very fast
# sum_mode=False, uses a std filtered mean approach which is much slower, but is
# very useful when it comes to cleaning the spectrum of noise. The number of std
# to include can be set by n_std_filter.
#
# Reference_arrays can be supplied or not. A list of (n,2) arrays can be given.
# If two datasets contain one of the two refference pair strings and are identical
# when those strings are removed, are paired up. The average of a dataset pair is 
# calculated as dataset_0 / dataset_1. Division is done before averaging.
#
# Examples pairs produced: 
# reference_arrays=['CH1', 'CH2']: ['Stressing/CH1 S1','Stressing/CH2 S1'], 
#                                  ['Stressing/CH1 !S1','Stressing/CH2 !S1']
# reference_arrays=['Stressing/CH1 S1','Stressing/CH2 S1'], 
#                  ['Stressing/CH1 !S1','Stressing/CH2 !S1']: 
#                                  ['Stressing/CH1 S1','Stressing/CH2 S1'], 
#                                  ['Stressing/CH1 !S1','Stressing/CH2 !S1']
api.compute_scan_avg(
	runnr,
	data_folder,
	arrays=['/'],
	sum_mode=False,
	reference_arrays=['CH1', 'CH2'],
)

# Compute a model assuming the first axis of the arrays is compatible with an
# angle measurement. 
# method='projection' uses trapezoidal projection
# method in ('lsq','lsq_robust') uses linear least-squares fit to requested harmonics
# It either does it for each eventnr or if supplied, it uses the scan_avg
#
# harmonics wanted can be tuned
# e.g. harmonics=(0,4,8,12) or any other combination of different integers
api.compute_model(
	runnr,
	data_folder,
	arrays=['Stressing'],
	use_scan_avg=True,
	method='lsq',
	harmonics=(0, 4, 8),
)
```

Example: single run with masking

```python
import udqs_ada as api

data_folder = "data_folder"
runnr = ['064']

# Create virtual dataset for the run (does not overwrite existing VDS)
api.create_virtual_file(runnr, data_folder, overwrite=False)

# Mask: keep only events with eventnr < 2500
#
# A mask can be generated on any dataset and any boolean expressing
# Example: mask={'scan_cycle': '(array > 1) * (array < 10)', 'scan_step': 'array == 10'}
mask = {'eventnr': '(array < 2500)'}
mask = api.mask_data(runnr, data_folder, mask)

# Compute reduced scan averages (per-scan step), use the mask and reference arrays
#
# It is set to now use tha mask computed
#
# Use 'LakeshoreTC/Temp. B (K)' as "scan" steps by binning the values. Bin size is
# inferred, but can be given by n_scan_bins
api.compute_scan_avg(
	runnr,
	data_folder,
	arrays=['/'],
	sum_mode=False,
	reference_arrays=['CH1', 'CH2'],
	use_mask=mask,
	scan_step_array='LakeshoreTC/Temp. B (K)'
)

# Compute model parameters from the reduced scan
api.compute_model(
	runnr,
	data_folder,
	['Stressing'],
	use_scan_avg=True,
	method='projection',
	harmonics=(0, 1, 2, 4, 8),
)
```

Example: multiple-run processing

```python
import udqs_ada as api

data_folder = "data_folder"
runnr = ['070', '071', '072', '073', '074']

# Build a concatenated VDS for the listed runs
api.create_virtual_file(runnr, data_folder, overwrite=False)

# Mask: only keep the first 3 scan cycles
mask = {'scan_cycle': '(array <= 3)'}
mask = api.mask_data(runnr, data_folder, mask)

# Compute reduced scan averages (per-scan step), use the mask and reference arrays
api.compute_scan_avg(
	runnr,
	data_folder,
	arrays=['/'],
	sum_mode=False,
	reference_arrays=['CH1 !S1', 'CH1 S1'],
	use_mask=mask,
)

# Compute model parameters from the reduced scan
api.compute_model(
	runnr,
	data_folder,
	['Stressing'],
	use_scan_avg=True,
	method='lsq',
	harmonics=(0, 4, 8),
)
```

Notes
-----
- The functions use multiprocessing with `spawn` start method and create
	temporary chunk files under `tmp_chunks_<run>`; these are removed when
	processing completes.
- Input `runnr` may be a single run id (string) or a sequence of run ids
	— when multiple runs are provided their VDS sources are concatenated.
- See the module docstrings and `api.py` for full parameter details and
	advanced options (e.g. `reference_arrays`, `scan_step_array`).

Requirements
------------
Dependencies are listed in `requirements.txt`. Key packages include:
`h5py`, `numpy`, `dask`, `psutil`, `tqdm`.

