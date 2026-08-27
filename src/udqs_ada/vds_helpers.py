import h5py as h5
import numpy as np
import os

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

def process_event_group(f, file, key, common_fields, vsource_lists):
    if 'scan_step' in f[key].attrs and 'scan_cycle' in f[key].attrs:
        scan_step = f[key].attrs["scan_step"]
        scan_cycle = f[key].attrs["scan_cycle"]
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

    for attr_key in f[key].attrs.keys():
        if attr_key.endswith('_mon_val'):
            attr_value = f[key].attrs[attr_key]
            if attr_key not in vsource_lists:
                vsource_lists[attr_key] = []
            vsource_lists[attr_key].append(np.array([attr_value], dtype=np.float64))
        elif attr_key.endswith('_set_val'):
            attr_value = f[key].attrs[attr_key]
            if attr_key not in vsource_lists:
                vsource_lists[attr_key] = []
            vsource_lists[attr_key].append(np.array([attr_value], dtype=np.float64))

    for group in f[key].keys():
        proc_time = f[key][group].attrs.get("proc_time", None)
        rep_rate = f[key][group].attrs.get("rep_rate", None)
        timestamp = f[key][group].attrs.get("timestamp", None)
        position = f[key][group].attrs.get("position", None)

        for field, value in zip(common_fields[4:], [proc_time, rep_rate, timestamp, position]):
            if value is None:
                continue
            if group+'/'+field not in vsource_lists:
                vsource_lists[group+'/'+field] = []
            if field in ['proc_time', 'rep_rate', 'position']:
                vsource_lists[group+'/'+field].append(np.array([value], dtype=np.float64))
            elif field == 'timestamp':
                vsource_lists[group+'/'+field].append(np.array([value], dtype=h5.string_dtype(encoding='utf-8')))

        for attr_key in f[key][group].attrs.keys():
            if group + '/' + attr_key not in vsource_lists:
                vsource_lists[group + '/' + attr_key] = []
            attr_value = f[key][group].attrs[attr_key]
            if isinstance(attr_value, (str, bytes)):
                vsource_lists[group + '/' + attr_key].append(np.array([attr_value], dtype=h5.string_dtype(encoding='utf-8')))
            else:
                vsource_lists[group + '/' + attr_key].append(np.array([attr_value], dtype=np.float64))

        for dataset in f[key][group].keys():
            if isinstance(f[key][group][dataset], h5.Dataset):
                dataset_path = f[key][group][dataset].name
                dataset_name = group + '/' + dataset_path.split("/")[-1]
                dataset_shape = f[key][group][dataset].shape
                dataset_dtype = f[key][group][dataset].dtype

                if dataset_name not in vsource_lists:
                    vsource_lists[dataset_name] = []

                vsource = h5.VirtualSource(os.path.abspath(file), dataset_path, shape=dataset_shape, dtype=dataset_dtype)
                vsource_lists[dataset_name].append(vsource)

def process_files(fileList, common_fields, last_event, vsource_lists):
    progress_counter = 0
    for file in fileList:
        with h5.File(file, 'r', swmr=True) as f:
            for key in f.keys():
                if key.startswith('Event_'):
                    process_event_group(f, file, key, common_fields, vsource_lists)
                    progress_counter += 1
                    progress_bar(progress_counter, last_event, prefix="Progress")

def _create_virtual_file(runnr, dataDir, fileprefix='File', overwrite=None):
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
                process_files(files_to_process, common_fields, last_event-first_event, vsource_lists)
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
        process_files(fileList, common_fields, last_event-first_event, vsource_lists)
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
