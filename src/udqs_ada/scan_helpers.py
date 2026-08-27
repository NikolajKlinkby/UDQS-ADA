import h5py as h5
import numpy as np
import os
import gc
import psutil
import dask.array as da

def first_diff_index_simple(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

def process_chunk(chunk_idx, chunk_indices, mask, arr, remaining_idx, virtual_file_path, tmp_dir, sum_mode, n_std_filter, counter, lock, batch_size=10):
    ref_arr_available = isinstance(arr, (list, tuple))
    if ref_arr_available:
        arr_name = arr[0] + '-' + arr[1][first_diff_index_simple(arr[0], arr[1]):]
    else:
        arr_name = arr
    
    safe_arr_name = arr_name.replace('/', '_')
    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx}_{safe_arr_name.replace('/','_')}.h5")
    if os.path.exists(tmp_path):
        with h5.File(tmp_path, 'r') as f_tmp:
            if f_tmp.attrs.get('done', False):
                return tmp_path, chunk_indices, arr_name
            
    with h5.File(virtual_file_path, 'r', rdcc_nbytes=1024*1024, rdcc_nslots=1000, rdcc_w0=0) as f_in, h5.File(tmp_path, 'a', rdcc_nbytes=1024*1024, rdcc_nslots=1000, rdcc_w0=0) as f_tmp:
        if sum_mode:
            mask[:,:remaining_idx] = 0
        
        if ref_arr_available:
            data = f_in[arr[0]]
            ref_data = f_in[arr[1]]
        else:
            data = f_in[arr]

        data_ndim = data.ndim
        _data_shape = data.shape
        _ref_shape = ref_data.shape if ref_arr_available else None

        if ref_arr_available:
            if _data_shape != _ref_shape:
                print(f"Warning: Shape mismatch between paired arrays '{arr[0]}' and '{arr[1]}'. Skipping chunk.")
                return tmp_path, chunk_indices, arr_name
            
        if data_ndim == 1 or 'scan_step_mask' in arr_name:
            return tmp_path, chunk_indices, arr_name

        chunk_shape = (1, _data_shape[1], _data_shape[2]) if data_ndim == 3 else (1, _data_shape[1])
        data_shape = (len(chunk_indices),) + _data_shape[1:]
        if sum_mode:
            sum_key = f"{arr_name}_sum"
            count_key = f"{arr_name}_count"
            f_tmp_sum = f_tmp.create_dataset(sum_key, shape=data_shape, dtype='f', chunks=chunk_shape)
            f_tmp_count = f_tmp.create_dataset(count_key, shape=(len(chunk_indices),), dtype='i')
        else:
            mean_key = f"{arr_name}_mean"
            f_tmp_mean = f_tmp.create_dataset(mean_key, shape=data_shape, dtype='f', chunks=chunk_shape)
            std_key = f"{arr_name}_std"
            f_tmp_std = f_tmp.create_dataset(std_key, shape=data_shape, dtype='f', chunks=chunk_shape)

        local_count = 0
        for idx, i in enumerate(chunk_indices):
            if sum_mode:
                indices = np.where(mask[i])[0]
                if len(indices) == 0:
                    step_val = np.zeros(_data_shape[1:])
                else:
                    step_val = np.sum(np.where(ref_data[indices]!=0, data[indices]/ref_data[indices], 0), axis=0) if ref_arr_available else np.sum(data[indices], axis=0)
                f_tmp_sum[idx, :] = step_val
                f_tmp_count[idx] = len(indices)
            else:
                indices = np.where(mask[i])[0]
                if len(indices) == 0:
                    mean_step = np.zeros(_data_shape[1:], dtype=float)
                    mean_std_step = np.zeros(_data_shape[1:], dtype=float)
                elif len(indices) == 1:
                    mean_step = np.array(np.where(ref_data[indices[0]]!=0, data[indices[0]]/ref_data[indices[0]], 0), dtype=float) if ref_arr_available else np.array(data[indices[0]], dtype=float)
                    mean_std_step = np.zeros_like(mean_step)
                else:
                    block = np.where(ref_data[indices]!=0, data[indices]/ref_data[indices], 0) if ref_arr_available else data[indices]
                    n_std = n_std_filter
                    median_step = np.median(block, axis=0, keepdims=True)
                    std_step = np.std(block, axis=0, keepdims=True)
                    # widen the acceptance window in steps until every element has at least one
                    # valid sample. Window is median +/- std * n_std. Increase n_std in
                    # small increments (0.1) until all elements have at least one
                    # valid sample or until a reasonable cap is reached.
                    max_n_std = 10.0
                    while n_std <= max_n_std:
                        delta = std_step * n_std
                        lower = median_step - delta
                        upper = median_step + delta
                        mask_ok = (block >= lower) & (block <= upper)
                        valid_counts = np.sum(mask_ok, axis=0)
                        if np.all(valid_counts > 0):
                            break
                        n_std += 0.1
                    # mask out outliers and compute sums/stats on the retained samples
                    filtered_block = np.where(mask_ok, block, np.nan)
                    sums = np.nansum(filtered_block, axis=0, dtype=float)
                    sumsq = np.nansum(filtered_block**2, axis=0, dtype=float)
                    total_count = np.sum(~np.isnan(filtered_block), axis=0)
                    # guard against divide-by-zero (shouldn't happen because of the while condition)
                    total_count = np.where(total_count == 0, 1, total_count)
                    mean_step = sums / total_count
                    mean_std_step = np.sqrt(np.maximum(sumsq / total_count - mean_step**2, 0))
                    del block, filtered_block, mask_ok, median_step, std_step, valid_counts, sums, sumsq, total_count
                f_tmp_std[idx, :] = mean_std_step
                f_tmp_mean[idx, :] = mean_step
            local_count += 1
            if local_count % batch_size == 0 or idx == len(chunk_indices) - 1:
                with lock:
                    counter.value += local_count
                local_count = 0
            if sum_mode:
                del step_val, indices
            else:
                del mean_step, mean_std_step, indices
            gc.collect()
        if sum_mode:
            f_tmp_sum.flush()
            f_tmp_count.flush()
            del f_tmp_sum, f_tmp_count, data
            if ref_arr_available:
                del ref_data
        else:
            f_tmp_mean.flush()
            f_tmp_std.flush()
            del f_tmp_mean, f_tmp_std, data
            if ref_arr_available:
                del ref_data
        f_in.flush()
        f_tmp.flush()
        gc.collect()

    with h5.File(tmp_path, 'a') as f_tmp:
        f_tmp.attrs['done'] = True
    return tmp_path, chunk_indices, arr_name

def dry_run_process_chunk(chunk_idx, chunk_indices, mask, arrays, remaining_events, virtual_file_path, tmp_dir, sum_mode, n_std_filter, batch_size=10):
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info().rss / (1024 ** 2)
    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx}.h5")
    for arr_name, remaining_idx in zip(arrays,remaining_events):
        if isinstance(arr_name, (list, tuple)):
            _arr_name = arr_name[0]
        else:
            _arr_name = arr_name
        with h5.File(virtual_file_path, 'r') as f_in, h5.File(tmp_path, 'a') as f_tmp:
            data = f_in[_arr_name]
            gc.collect()
            _mem = proc.memory_info().rss / (1024 ** 2)
            if _mem > mem:
                mem = _mem
        if isinstance(arr_name, (list, tuple)):
            with h5.File(virtual_file_path, 'r') as f_in, h5.File(tmp_path, 'a') as f_tmp:
                data = f_in[arr_name[1]]
                gc.collect()
                _mem = proc.memory_info().rss / (1024 ** 2)
                if _mem > mem:
                    mem = _mem
    return mem

def find_dataset_containing(h5group, substring):
    if substring[0] == '/':
        substring = substring[1:]
    matches = []
    def visitor(name, obj):
        if isinstance(obj, h5.Dataset) and substring in name and ('_sum' in name or '_count' in name):
            matches.append(name)
    h5group.visititems(visitor)
    return matches

def iterate_datasets(h5_group, path=""):
    for key in h5_group.keys():
        item = h5_group[key]
        item_path = f"{path}/{key}" if path else key
        if isinstance(item, h5.Dataset):
            yield item_path, item
        elif isinstance(item, h5.Group):
            yield from iterate_datasets(item, path=item_path)
