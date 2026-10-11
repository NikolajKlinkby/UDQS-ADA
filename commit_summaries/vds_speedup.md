# `create_virtual_file`: faster event reading and optional parallel reading

This change makes building a virtual file about three times faster in a single process and adds
an `n_workers` argument that reads the raw files in parallel. The virtual files it produces are
identical to the previous code's.

## What changed

### 1. Each HDF5 object is opened once per event

Building a virtual file is linear in the number of events, and 94% of the time was spent in
`process_event_group`. That function looked the same groups and datasets up by name again and
again (`f[key][group][dataset]...`), about 84 lookups per event. It now opens the event group,
each controller group and each dataset once, and reads each group's attributes in one pass.

### 2. `n_workers`: read the raw files in parallel

`create_virtual_file(..., n_workers=N)` reads the raw files with `N` processes, one file per
process, and merges the results in file order. The default is `n_workers=1`, which reads the
files one after the other as before.

```python
api.create_virtual_file(runnr, data_folder, overwrite=False, n_workers=4)
```

Workers are started with the `spawn` method so they inherit no open HDF5 files and behave the
same on Linux, macOS and Windows. As with `compute_scan_avg`, a script that uses more than one
worker must make the call inside `if __name__ == '__main__':`. If a worker dies, the error
message says so.

## Measurements

Synthetic runs with 2 cycles, 2 controller groups and 4 datasets per event, 2,000 events per raw
file, on a 2-core machine with 8 GB of memory. Timings varied by up to about 25% between
repeated runs, so compare within a table.

Single process, before and after change 1:

| Steps  | Events | Original | Patched |
|-------:|-------:|---------:|--------:|
|    500 |  1,000 |    4.8 s |   1.7 s |
|  4,000 |  8,000 |   44.5 s |  14.0 s |
|  8,000 | 16,000 |   77.2 s |  26.8 s |

Patched code with parallel reading:

| Steps  | Events | Files | 1 worker | 2 workers | 4 workers |
|-------:|-------:|------:|---------:|----------:|----------:|
|  8,000 | 16,000 |     8 |   34.5 s |    22.3 s |    25.3 s |
| 15,000 | 30,000 |    15 |   80.5 s |    46.0 s |    44.8 s |

The test machine has 2 cores, so 4 workers bring nothing here. Reading the files is about 85%
of the single-process time at 15,000 steps; the rest (assembling the virtual datasets, about
11 s) is not parallel and sets the floor on machines with more cores.

## Correctness

Virtual files were compared object by object: names, order, shapes, data types, values,
attributes, and the source file and dataset behind every entry of every virtual dataset. They
were identical:

- between the original code and the patched code with 1 worker, for a fresh build
- between 1, 2 and 4 workers
- between the original code and 3 workers for a fresh build, an incremental update after new
  files were added (`overwrite=False`), and two runs combined

A full run through `compute_scan_avg` and `compute_model` on a virtual file built with 2 workers
gave the same reduced and model files as the original code.

## Behaviour and compatibility

- **Results are unchanged**, for any number of workers.
- **The default is unchanged**: `n_workers=1`.
- **Gain depends on the number of raw files.** A run stored in a single file gains nothing from
  more workers.

## Not changed

Two existing problems were noticed and deliberately left as they are, because fixing them
changes behaviour or output:

- For each controller, the `proc_time`, `rep_rate`, `timestamp` and `position` datasets get two
  entries per event. `process_event_group` appends these four attributes once by name and once
  more in its loop over all attributes. This patch reproduces that exactly.
- When `runnr` is a single string, `create_virtual_file` passes `overwrite=None` on and always
  asks interactively, whatever `overwrite` was given. A list such as `['070']` works as
  documented.

## Files

- `src/udqs_ada/vds_helpers.py`: `process_event_group` rewritten, `_read_file` added,
  `process_files` and `_create_virtual_file` take `n_workers`
- `src/udqs_ada/api.py`: `create_virtual_file` takes `n_workers`
