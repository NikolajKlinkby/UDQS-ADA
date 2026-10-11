# `compute_scan_avg`: event lists, batched reads, no per-step garbage collection



Comment by the human here: I noticed that, especially for combined scans over several scanningparameters, the size of the virtual dataset would be comparable or even larger than the raw data. The execution time was becoming a bottleneck as well. With the most recent scan over 20 hours, running the script took around an hour, and produced faulty results on my laptop. So after a bit of back and forth, and claude working, writing and benchmarking for an hour and a half, it came up with this. Will test it now on the running scan.



This change makes `compute_scan_avg` faster and lighter on long scans without changing its
results. On a synthetic 15,000-step scan it ran in 54 s where the previous code is estimated to
need 45 to 50 minutes, used about half the memory, and no longer grows the virtual file.

## What changed

### 1. Per-step event lists replace the `scan_step_mask` table

Before, the events of each scan step were looked up in a boolean table of shape
`(n_steps, n_events)`. The table has exactly one `True` per event, so it carries no more
information than the `scan_step` dataset, but it takes `n_steps * n_events` bytes. It was built
with a per-step dask loop, written uncompressed into the virtual file, and sent in full to every
worker task.

Now the events are kept grouped by step in two integer arrays (`ScanGroups` in
`mask_helpers.py`):

- `order`: event numbers grouped by step, ascending within a step
- `starts`: where each step's group begins in `order`

The events of step `i` are `order[starts[i]:starts[i+1]]`. Building this is one stable sort of
`scan_step`. Each worker task receives only the event numbers of its own steps. Nothing large is
stored in the virtual file, and the grouping is recomputed on every call, so it cannot be out of
date with the events in the file.

### 2. No `gc.collect()` after every scan step

The worker loop called a full garbage collection after each step. That took about 36 ms per call
against about 1.5 ms for the step's actual work (94% of a worker's time in a profile). The arrays
of a step are freed as soon as they are deleted, so the call is not needed. One collection per
chunk remains.

### 3. Events are read for many steps at once

Every read from a virtual dataset costs time proportional to the number of source files and
events in the run. One read per step therefore made the total time grow with the square of the
run length. Workers now read the events of a batch of steps in one call (up to
`READ_BLOCK_BYTES`, 64 MB per array, always at least one whole step) and split the block by step
in memory. Contiguous events are read as slices.

## Measurements

Synthetic runs: 2 cycles, 4 arrays of 180 values plus scalars, `arrays=['/']`,
`sum_mode=False`, 4 workers requested, on a 2-core machine with 8 GB of memory. Absolute numbers
will differ on other machines and data.

Wall time of `compute_scan_avg` in seconds:

|  Steps | Events | Original | Change 1 | Changes 1+2 | Changes 1+2+3 |
| -----: | -----: | -------: | -------: | ----------: | ------------: |
|    500 |  1,000 |       61 |       61 |           9 |             9 |
|  2,000 |  4,000 |      231 |      225 |          28 |            14 |
|  4,000 |  8,000 |      496 |      452 |          72 |            21 |
|  8,000 | 16,000 |  not run |  not run |         227 |            33 |
| 15,000 | 30,000 |  not run |  not run |         736 |            54 |

The original was not run at the two largest sizes because of the time it would take.
Extrapolating its three measured sizes gives roughly 45 to 50 minutes at 15,000 steps.

Peak memory of all processes together, in MB. "Table" is the previous table-based design with
the per-step collection removed and the table built with numpy, so it could be run at the large
sizes:

|  Steps | Table | Event lists |
| -----: | ----: | ----------: |
|  4,000 | 1,569 |       1,109 |
|  8,000 | 2,380 |       1,599 |
| 15,000 | 4,646 |       2,456 |

At 15,000 steps the table version also reduced itself from 4 workers to 3 to fit in memory.

Size of the virtual file after averaging: unchanged with this patch (12.5 MB at 4,000 steps,
48 MB at 15,000). Before, it grew by `n_steps * n_events` bytes (44.5 MB at 4,000 steps; about
500 MB at 15,000).

## Correctness

The reduced files (and the model file where one was computed) were compared dataset by dataset,
including attributes, between the original and the patched code. They were identical in:

- `sum_mode=True`, followed by `compute_model`
- `sum_mode=False` with `reference_arrays` and `n_std_filter`
- a user mask from `mask_data`
- binned steps via `scan_step_array` / `n_scan_bins`
- two runs combined
- a resumed `sum_mode=True` run after new files were added
- every benchmark size where both versions ran

The event lists were also checked against the original table on random inputs (events outside
the scan, gaps in the step numbers, 1D and 2D user masks, resumed runs, binned steps), and the
batched reads against per-step reads for several block sizes.

## Behaviour and compatibility

- **Results are unchanged.**
- **`memory_fraction` is no longer used by `compute_scan_avg`.** It only set the dask chunk size
  for building the table. The argument is still accepted. `mask_data` still uses it.
- **`scan_step_mask` is no longer written to the virtual file.** A table left in an existing
  virtual file is ignored. To keep the binned mode's "reuse the previous number of steps"
  behaviour, the number of steps of the last grouping is stored as the file attribute
  `scan_step_rows`.
- **`get_scan_step_mask` still exists** for code that calls it directly and returns the same
  table, built from the event lists. It no longer caches the table in the file.
- **Worker memory estimate.** The dry run used to measure an idle worker that already held a
  copy of the table, so the estimate tracked the table size and ignored the data being read. It
  is now twice the idle worker size, plus the read blocks, plus six times the largest scan step.
- **`use_mask`** accepts a per-event array as before, and also a `(n_steps, n_events)` array,
  which the old elementwise product allowed.

## Not changed

- Memory now scales mainly with the number of events, because every process loads the virtual
  file's list of sources. This patch does not address that.
- Passing a group name such as `arrays=['Caen_DT5730SB']` to `compute_scan_avg` raises
  "Invalid array/group path" before and after this patch. Dataset paths and `'/'` work.
- The merge of the temporary chunk files and `compute_model` are untouched.

## Files

- `src/udqs_ada/mask_helpers.py`: `ScanGroups`, `group_events`, `get_scan_step_groups`;
  `get_scan_step_mask` kept as a thin wrapper
- `src/udqs_ada/scan_helpers.py`: `read_events`, batched reads and event lists in
  `process_chunk`, per-step `gc.collect()` removed
- `src/udqs_ada/api.py`: `compute_scan_avg` uses the event lists; worker memory estimate
