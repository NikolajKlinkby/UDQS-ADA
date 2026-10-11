import h5py as h5
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

def progress_bar(iteration, total, prefix='', length=50, every=100):
    if iteration % every == 0 or iteration == total:
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        import sys
        sys.stdout.write(f'\r{prefix} |{bar}| {iteration}/{total}')
        sys.stdout.flush()
    if iteration == total:
        print()

def normalize_to_bytes(arr):
    arr = np.asarray(arr, dtype=object)
    out = []
    for x in arr:
        if isinstance(x, str):
            out.append(x.encode("utf-8"))
        else:
            out.append(x)
    return np.array(out, dtype=object)

def get_consecutive_files(fileList, fileprefix):
    import h5py as h5
    numbered_files = []
    for f in fileList:
        try:
            num = int(f.split('-')[-1].split('.')[0])
            numbered_files.append((num, f))
        except ValueError:
            continue

    numbered_files.sort(key=lambda x: x[0])

    consecutive_files = []
    expected = 1
    for num, f in numbered_files:
        if num == expected:
            try:
                with h5.File(f, 'r', swmr=True) as h:
                    _ = list(h.keys())
                consecutive_files.append(f)
                expected += 1
            except Exception as e:
                print(f"Skipping incomplete file {f}: {e}")
                break
        else:
            print(f"Missing file number {expected}, stopping at {num}")
            break

    return consecutive_files

def collect_from_meta(meta_grp):
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5.Group):
            full_path = name.replace("\\", "/")
            if all(k in obj for k in ["filenames", "dset_names", "shapes", "dtypes"]):
                datasets.append(name)

    meta_grp.visititems(visitor)
    return datasets

def list_h5_datasets(h5_file, path="/"):
    datasets = []

    def visitor(name):
        full_path = os.path.join(path.strip("/"), name).replace("\\", "/")
        if not full_path.startswith("/"):
            full_path = "/" + full_path
        obj_class = h5_file.get(full_path, getclass=True)
        if obj_class == h5.Dataset and 'vds_metadata' not in full_path:
            datasets.append(full_path)

    h5_file[path].visit(visitor)
    return datasets

def process_event_group(f, file, key, common_fields, vsource_lists, abs_file=None):
    # Every HDF5 object is opened once and kept in a local variable. Looking a group or dataset
    # up by name again and again (f[key][group][dataset] ...) was most of the time spent here.
    if abs_file is None:
        abs_file = os.path.abspath(file)
    event = f[key]
    event_attrs = dict(event.attrs.items())

    if 'scan_step' in event_attrs and 'scan_cycle' in event_attrs:
        scan_step = event_attrs["scan_step"]
        scan_cycle = event_attrs["scan_cycle"]
    else:
        scan_step = -1
        scan_cycle = -1
    filenr = int(file.split('-')[-1].split('.')[0])
    eventnr = int(key.split("_")[1])

    for field, value in zip(common_fields[:4], [scan_step, scan_cycle, filenr, eventnr]):
        if field not in vsource_lists:
            vsource_lists[field] = []
        if field in ['scan_step', 'scan_cycle', 'filenr', 'eventnr']:
            vsource_lists[field].append(np.array([value], dtype=np.int32))

    for attr_key, attr_value in event_attrs.items():
        if attr_key.endswith('_mon_val'):
            if attr_key not in vsource_lists:
                vsource_lists[attr_key] = []
            vsource_lists[attr_key].append(np.array([attr_value], dtype=np.float64))
        elif attr_key.endswith('_set_val'):
            if attr_key not in vsource_lists:
                vsource_lists[attr_key] = []
            vsource_lists[attr_key].append(np.array([attr_value], dtype=np.float64))

    for group in event.keys():
        grp = event[group]
        grp_attrs = dict(grp.attrs.items())
        proc_time = grp_attrs.get("proc_time", None)
        rep_rate = grp_attrs.get("rep_rate", None)
        timestamp = grp_attrs.get("timestamp", None)
        position = grp_attrs.get("position", None)

        for field, value in zip(common_fields[4:], [proc_time, rep_rate, timestamp, position]):
            if value is None:
                continue
            if group+'/'+field not in vsource_lists:
                vsource_lists[group+'/'+field] = []
            if field in ['proc_time', 'rep_rate', 'position']:
                vsource_lists[group+'/'+field].append(np.array([value], dtype=np.float64))
            elif field == 'timestamp':
                vsource_lists[group+'/'+field].append(np.array([value], dtype=h5.string_dtype(encoding='utf-8')))

        for attr_key, attr_value in grp_attrs.items():
            if group + '/' + attr_key not in vsource_lists:
                vsource_lists[group + '/' + attr_key] = []
            if isinstance(attr_value, (str, bytes)):
                vsource_lists[group + '/' + attr_key].append(np.array([attr_value], dtype=h5.string_dtype(encoding='utf-8')))
            else:
                vsource_lists[group + '/' + attr_key].append(np.array([attr_value], dtype=np.float64))

        for dataset in grp.keys():
            obj = grp[dataset]
            if isinstance(obj, h5.Dataset):
                dataset_path = obj.name
                dataset_name = group + '/' + dataset_path.split("/")[-1]
                dataset_shape = obj.shape
                dataset_dtype = obj.dtype

                if dataset_name not in vsource_lists:
                    vsource_lists[dataset_name] = []

                vsource = h5.VirtualSource(abs_file, dataset_path, shape=dataset_shape, dtype=dataset_dtype)
                vsource_lists[dataset_name].append(vsource)

def _read_file(args):
    """Worker for process_files: read one raw file and return what it contributes, in a form
    that can be sent between processes. Virtual sources are returned as
    (dataset path, shape, dtype) and rebuilt in the parent."""
    file, common_fields = args
    abs_file = os.path.abspath(file)
    lists = {}
    n_events = 0
    with h5.File(file, 'r', swmr=True) as f:
        for key in f.keys():
            if key.startswith('Event_'):
                process_event_group(f, file, key, common_fields, lists, abs_file)
                n_events += 1
    out = []
    for name, items in lists.items():
        if items and isinstance(items[0], h5.VirtualSource):
            out.append((name, True, [(vs.name, vs.shape, vs.dtype) for vs in items]))
        else:
            out.append((name, False, items))
    return abs_file, n_events, out


def process_files(fileList, common_fields, last_event, vsource_lists, n_workers=1):
    """Read every event of every file into vsource_lists.

    With n_workers > 1 the files are read in parallel, one process per file, and merged in file
    order, which gives exactly the same result as reading them one after the other."""
    n_workers = max(1, min(int(n_workers or 1), len(fileList)))
    progress_counter = 0
    if n_workers == 1:
        for file in fileList:
            abs_file = os.path.abspath(file)
            with h5.File(file, 'r', swmr=True) as f:
                for key in f.keys():
                    if key.startswith('Event_'):
                        process_event_group(f, file, key, common_fields, vsource_lists, abs_file)
                        progress_counter += 1
                        progress_bar(progress_counter, last_event, prefix="Progress")
        return

    # 'spawn' so the workers start clean on every platform and inherit no open HDF5 files
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context('spawn')) as pool:
            for abs_file, n_events, lists in pool.map(_read_file, [(file, common_fields) for file in fileList]):
                for name, is_source, items in lists:
                    target = vsource_lists.setdefault(name, [])
                    if is_source:
                        target.extend(h5.VirtualSource(abs_file, path, shape=shape, dtype=dtype) for path, shape, dtype in items)
                    else:
                        target.extend(items)
                progress_counter += n_events
                progress_bar(progress_counter, last_event, prefix="Progress")
    except BrokenProcessPool as e:
        raise RuntimeError(
            "Reading the files in parallel failed because a worker process died. If this is a "
            "script, the calls must be inside `if __name__ == '__main__':` (as for "
            "compute_scan_avg); otherwise use n_workers=1."
        ) from e

def _create_virtual_file(runnr, dataDir, fileprefix='File', overwrite=None, n_workers=1):
    """
    Internal helper that builds (or updates) the virtual dataset file for a
    single run. This inspects HDF5 files in `dataDir/Run-<runnr>`, identifies
    per-file datasets and Event_* groups, and constructs a
    `Run-<runnr>_VirtualData.h5` file containing VirtualSources and simple
    concatenated datasets for scan values and other 1D arrays.

    Parameters:
    - runnr: str
        Single run identifier.
    - dataDir: str
        Path to the directory that contains `Run-<runnr>`.
    - fileprefix: str, optional
        Prefix to identify data files inside the run directory.
    - overwrite: bool or None, optional
        Controls behavior when the target virtual file already exists.
    - n_workers: int, optional
        Number of processes used to read the raw files (default 1).

    Returns:
    - int
        1 when the virtual file was created/updated, 0 if nothing changed or
        no valid files were found.
    """
    fullName = os.path.join(dataDir, 'Run-' + runnr)
    if not os.path.exists(fullName):
        exit(f"Directory {fullName} does not exist.")
    fileList = [os.path.join(fullName, file) 
            for file in os.listdir(fullName) 
            if file.startswith(fileprefix) and file.endswith('.h5')]
    fileList = get_consecutive_files(fileList, fileprefix)
    if not fileList:
        print("No valid files found.")
        return 0
    last_event = 0
    with h5.File(fileList[-1], 'r') as f:
        for key in f.keys():
            if key.startswith('Event_'):
                eventnr = int(key.split('_')[1])
                if eventnr > last_event:
                    last_event = eventnr
    first_event = 1e+32
    with h5.File(fileList[0], 'r') as f:
        for key in f.keys():
            if key.startswith('Event_'):
                eventnr = int(key.split('_')[1])
                if eventnr < first_event:
                    first_event = eventnr
    first_event -= 1
    vds_file = os.path.join(dataDir, 'Run-' + runnr + '_VirtualData.h5')
    if os.path.exists(vds_file):
        if overwrite is None:
            response = input(f"File '{vds_file}' already exists. Overwrite? (y/n): ").strip().lower()
            overwrite = response == 'y'
        if not overwrite:
            print("Checking for missing files in the existing virtual dataset...")
            with h5.File(vds_file, 'r') as vds:
                if 'filenr' in vds:
                    existing_file_numbers = set(vds['filenr'][:])
                else:
                    existing_file_numbers = set()
            file_numbers_to_process = set()
            for file in fileList:
                filenr = int(file.split('-')[-1].split('.')[0])
                file_numbers_to_process.add(filenr)
            missing_file_numbers = file_numbers_to_process - existing_file_numbers
            files_to_process = [file for file in fileList if int(file.split('-')[-1].split('.')[0]) in missing_file_numbers]
            if not files_to_process:
                print("All files are already loaded. No missing files.")
                return 0
            else:
                print(f"Processing missing files: {len(files_to_process)}")
                first_event = 1e+32
                with h5.File(files_to_process[0], 'r') as f:
                    for key in f.keys():
                        if key.startswith('Event_'):
                            eventnr = int(key.split('_')[1])
                            if eventnr < first_event:
                                first_event = eventnr
                first_event -= 1
                vsource_lists = {}
                common_fields = ['scan_step', 'scan_cycle', 'filenr', 'eventnr', 'proc_time', 'rep_rate', 'timestamp', 'position']
                process_files(files_to_process, common_fields, last_event-first_event, vsource_lists, n_workers=n_workers)
                scan_val_dict = {}
                with h5.File(vds_file, 'a') as vds:
                    for dataset_name, vsource_list in vsource_lists.items():
                        if isinstance(vsource_list[0], np.ndarray):
                            vds_data = np.concatenate(vsource_list)
                            if dataset_name in vds:
                                existing_data = vds[dataset_name][:]
                                if len(existing_data) > 0 and isinstance(existing_data[0], (bytes, bytearray)):
                                    vds_data = normalize_to_bytes(vds_data)
                                vds_data = np.concatenate([existing_data, vds_data])
                                del vds[dataset_name]
                            if 'scan_step' in dataset_name or 'set_val' in dataset_name:
                                scan_val_dict[dataset_name] = vds_data
                            vds.create_dataset(dataset_name, data=vds_data)
                        else:
                            if dataset_name in vds:
                                ds_grp = vds["vds_metadata"][dataset_name]
                                filenames = [name.decode('utf-8') for name in ds_grp["filenames"][:]]
                                dset_names = [name.decode('utf-8') for name in ds_grp["dset_names"][:]]
                                shapes = [tuple(shape) for shape in ds_grp["shapes"][:]]
                                dtypes = [np.dtype(dt.decode('utf-8')) for dt in ds_grp["dtypes"][:]]
                                old_sources = []
                                for fn, dset, shape, dt in zip(filenames, dset_names, shapes, dtypes):
                                    vs = h5.VirtualSource(fn, dset, shape=shape, dtype=dt)
                                    old_sources.append(vs)
                                del vds[dataset_name]
                                vsource_list = old_sources + vsource_list
                            layout_shape = (len(vsource_list),) + vsource_list[0].shape
                            layout = h5.VirtualLayout(shape=layout_shape, dtype=vsource_list[0].dtype)
                            for idx, vsource in enumerate(vsource_list):
                                layout[idx] = vsource
                            vds.create_virtual_dataset(dataset_name, layout)
                            meta_grp = vds.require_group("vds_metadata")
                            filenames = np.array([vs.path.encode('utf-8') for vs in vsource_list])
                            dset_names = np.array([vs.name.encode('utf-8') for vs in vsource_list])
                            shapes = np.array([vs.shape for vs in vsource_list], dtype=np.int64)
                            dtypes = np.array([str(vs.dtype).encode('utf-8') for vs in vsource_list])
                            if dataset_name in meta_grp:
                                del meta_grp[dataset_name]
                            ds_grp = meta_grp.create_group(dataset_name)
                            ds_grp.create_dataset("filenames", data=filenames)
                            ds_grp.create_dataset("dset_names", data=dset_names)
                            ds_grp.create_dataset("shapes", data=shapes)
                            ds_grp.create_dataset("dtypes", data=dtypes)
                            
                    if 'scan_step' in scan_val_dict:
                        val, idx = np.unique(scan_val_dict['scan_step'], return_index=True)
                        idx = idx[val != -1]
                        
                        for key in scan_val_dict.keys():
                            if 'scan_step' not in key:
                                if 'step_'+ key in vds:
                                    del vds['step_'+key]
                                vds.create_dataset('step_'+key, data=scan_val_dict[key][idx])
                print(f"Missing files loaded into the virtual dataset: {vds_file}")
                return 1
        else:
            print(f"Overwriting existing virtual dataset: {vds_file}")
            os.remove(vds_file)
            print(f"Removed existing virtual dataset: {vds_file}")
    print(f"Creating new virtual dataset at: {vds_file}")
    with h5.File(vds_file, 'w') as vds:
        vsource_lists = {}
        common_fields = ['scan_step', 'scan_cycle', 'filenr', 'eventnr', 'proc_time', 'rep_rate', 'timestamp', 'position']
        process_files(fileList, common_fields, last_event-first_event, vsource_lists, n_workers=n_workers)
        scan_val_dict = {}
        for dataset_name, vsource_list in vsource_lists.items():
            if isinstance(vsource_list[0], np.ndarray):
                vds_data = np.concatenate(vsource_list)
                if dataset_name in vds:
                    existing_data = vds[dataset_name][:]
                    if len(existing_data) > 0 and isinstance(existing_data[0], (bytes, bytearray)):
                        vds_data = normalize_to_bytes(vds_data)
                    vds_data = np.concatenate([existing_data, vds_data])
                    del vds[dataset_name]
                if 'scan_step' in dataset_name or 'set_val' in dataset_name:
                    scan_val_dict[dataset_name] = vds_data
                vds.create_dataset(dataset_name, data=vds_data)
            else:
                if dataset_name in vds:
                    ds_grp = vds["vds_metadata"][dataset_name]
                    filenames = [name.decode('utf-8') for name in ds_grp["filenames"][:]]
                    dset_names = [name.decode('utf-8') for name in ds_grp["dset_names"][:]]
                    shapes = [tuple(shape) for shape in ds_grp["shapes"][:]]
                    dtypes = [np.dtype(dt.decode('utf-8')) for dt in ds_grp["dtypes"][:]]
                    old_sources = []
                    for fn, dset, shape, dt in zip(filenames, dset_names, shapes, dtypes):
                        vs = h5.VirtualSource(fn, dset, shape=shape, dtype=dt)
                        old_sources.append(vs)
                    del vds[dataset_name]
                    vsource_list = old_sources + vsource_list
                layout_shape = (len(vsource_list),) + vsource_list[0].shape
                layout = h5.VirtualLayout(shape=layout_shape, dtype=vsource_list[0].dtype)
                for idx, vsource in enumerate(vsource_list):
                    layout[idx] = vsource
                vds.create_virtual_dataset(dataset_name, layout)
                meta_grp = vds.require_group("vds_metadata")
                filenames = np.array([vs.path.encode('utf-8') for vs in vsource_list])
                dset_names = np.array([vs.name.encode('utf-8') for vs in vsource_list])
                shapes = np.array([vs.shape for vs in vsource_list], dtype=np.int64)
                dtypes = np.array([str(vs.dtype).encode('utf-8') for vs in vsource_list])
                if dataset_name in meta_grp:
                    del meta_grp[dataset_name]
                ds_grp = meta_grp.create_group(dataset_name)
                ds_grp.create_dataset("filenames", data=filenames)
                ds_grp.create_dataset("dset_names", data=dset_names)
                ds_grp.create_dataset("shapes", data=shapes)
                ds_grp.create_dataset("dtypes", data=dtypes)
        if 'scan_step' in scan_val_dict:
            val, idx = np.unique(scan_val_dict['scan_step'], return_index=True)
            idx = idx[val != -1]
            
            for key in scan_val_dict.keys():
                if 'scan_step' not in key:
                    if 'step_'+ key in vds:
                        del vds['step_'+key]
                        
                    vds.create_dataset('step_'+key, data=scan_val_dict[key][idx])
    return 1
