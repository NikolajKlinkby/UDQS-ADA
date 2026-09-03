import h5py as h5
import numpy as np
import os
import dask.array as da
from scipy.optimize import least_squares

def process_model_chunk(
    chunk_idx,
    chunk_indices,
    arr_name,
    angles,
    virtual_file_path,
    tmp_dir,
    counter,
    lock,
    batch_size=10,
    method='projection',
    harmonics=(0,4,8),
    reg=1e-8,
    weights=None,
    robust=False,
    use_scan_avg=False,
    reduced_scan_path=None,
    angle_index=1,
):
    safe_arr_name = arr_name.replace('/', '_')
    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx}_{safe_arr_name.replace('/','_')}.h5")
    # determine number of coefficients from requested harmonics
    A, keys = build_design_matrix(angles, harmonics)
    n_coeffs = len(keys)

    # ensure tmp dir exists
    os.makedirs(tmp_dir, exist_ok=True)

    # If using pre-averaged scan data, read from reduced file; otherwise read from virtual VDS
    if use_scan_avg:
        # try to infer reduced_scan_path if not provided
        if reduced_scan_path is None:
            reduced_scan_path = virtual_file_path.replace('_VirtualData.h5', '_ReducedScan.h5')
            if not os.path.exists(reduced_scan_path):
                reduced_scan_path = os.path.join(os.path.dirname(virtual_file_path), 'ReducedScan.h5')

        with h5.File(reduced_scan_path, 'r') as f_red, h5.File(tmp_path, 'a', rdcc_nbytes=1024*1024, rdcc_nslots=1000, rdcc_w0=0) as f_tmp:
            # try candidate names: arr_name, arr_name + '_mean', arr_name + '_sum'
            candidates = [arr_name, f"{arr_name}_mean", f"{arr_name}_sum"]
            dset_red = None
            for c in candidates:
                if c in f_red:
                    dset_red = f_red[c]
                    break

            if dset_red is None:
                raise KeyError(f"Could not find reduced dataset for '{arr_name}' in {reduced_scan_path}; tried: {candidates}")

            # infer last-dimension size (pairs) from reduced dataset
            shape_red = dset_red.shape
            ndim_red = len(shape_red)
            item_axis = 0

            if not (0 <= angle_index < ndim_red):
                raise ValueError(
                    f"angle_index={angle_index} is out of range for dataset '{arr_name}' "
                    f"with shape {shape_red}"
                )
            if angle_index == item_axis:
                raise ValueError(
                    f"angle_index cannot be {item_axis}; that axis is used to index chunk items"
                )

            angle_axis_in_slice = angle_index - 1
 
            rest_shape = tuple(
                s for ax, s in enumerate(shape_red) if ax not in (item_axis, angle_index)
            )
 
            chunk = (1, n_coeffs) + rest_shape
            new_shape = (len(chunk_indices), n_coeffs) + rest_shape
            dset_m = f_tmp.create_dataset(arr_name, shape=new_shape, dtype='f', chunks=chunk)
            
            local_count = 0
            for idx, i in enumerate(chunk_indices):
                arr_avg = np.moveaxis(dset_red[i], angle_axis_in_slice, 0)
                coef = get_model(arr_avg, angles, method=method, harmonics=harmonics, reg=reg, weights=weights, robust=robust)
                # ensure returned coef has shape (n_coeffs, pairs) or (n_coeffs,)
                coef = np.asarray(coef)
                dset_m[idx] = coef
                local_count += 1
                if local_count % batch_size == 0 or idx == len(chunk_indices) - 1:
                    with lock:
                        counter.value += local_count
                    local_count = 0
    else:
        with h5.File(virtual_file_path, 'r', rdcc_nbytes=1024*1024, rdcc_nslots=1000, rdcc_w0=0) as f_in, h5.File(tmp_path, 'a', rdcc_nbytes=1024*1024, rdcc_nslots=1000, rdcc_w0=0) as f_tmp:
            dset = f_in[arr_name]
            shape = dset.shape
            ndim = len(shape)
            item_axis = 0
            
            if not (0 <= angle_index < ndim):
                raise ValueError(
                    f"angle_index={angle_index} is out of range for dataset '{arr_name}' "
                    f"with shape {shape}"
                )
            if angle_index == item_axis:
                raise ValueError(
                    f"angle_index cannot be {item_axis}; that axis is used to index chunk items"
                )

            angle_axis_in_slice = angle_index - 1
    
            rest_shape = tuple(
                s for ax, s in enumerate(shape) if ax not in (item_axis, angle_index)
            )

            chunk = (1,) + shape[1:]
            darr = da.from_array(dset, chunks=chunk)
    
            chunk = (1, n_coeffs) + rest_shape
            new_shape = (len(chunk_indices), n_coeffs) + rest_shape
            dset_m = f_tmp.create_dataset(arr_name, shape=new_shape, dtype='f', chunks=chunk)

            local_count = 0
            for idx, i in enumerate(chunk_indices):
                data = np.moveaxis(darr[i].compute(), angle_axis_in_slice, 0)
                coef = get_model(data, angles, method=method, harmonics=harmonics, reg=reg, weights=weights, robust=robust)
                coef = np.asarray(coef)
                dset_m[idx] = coef
                local_count += 1
                if local_count % batch_size == 0 or idx == len(chunk_indices) - 1:
                    with lock:
                        counter.value += local_count
                    local_count = 0

    return tmp_path, chunk_indices, arr_name

def build_design_matrix(angles, harmonics):
    """
    Build design matrix A for harmonic regression.
    angles: 1D array (n_angles,) in radians
    harmonics: iterable of ints (e.g. [0,1,2,4,8])
      - if 0 in harmonics -> column of ones (DC)
      - for k>0 -> columns [cos(k*theta), sin(k*theta)]
    Returns A (n_angles, n_cols) and a list of keys describing columns in order.
    """
    thetas = np.asarray(angles)
    cols = []
    keys = []
    if 0 in harmonics:
        cols.append(np.ones_like(thetas))
        keys.append('I0')
    for k in harmonics:
        if k == 0:
            continue
        cols.append(np.cos(k * thetas))
        keys.append(f'I{int(k)}c')
        cols.append(np.sin(k * thetas))
        keys.append(f'I{int(k)}s')
    A = np.vstack(cols).T  # shape (n_angles, n_cols)
    return A, keys

def robust_ridge_irls(A, Y, coef, reg=1e-8, weights=None,
                      max_iter=8, tol=1e-6):
    """
    Robust ridge regression using vectorized IRLS.

    Parameters
    ----------
    A : (m,p)
        Design matrix.
    Y : (m,n)
        One column per spectrum.
    coef : (p,n)
        Initial ridge solution.
    reg : float
    weights : (m,) or None
        Optional measurement weights.
    """

    m, p = A.shape
    n = Y.shape[1]

    if weights is None:
        meas_w = np.ones(m)
    else:
        meas_w = np.asarray(weights)

    I = np.eye(p)

    for _ in range(max_iter):

        # --------------------------------------------------------
        # Residuals for ALL spectra
        # --------------------------------------------------------

        R = A @ coef - Y           # (m,n)
        R *= meas_w[:, None]

        # --------------------------------------------------------
        # Robust scale estimate (MAD)
        # --------------------------------------------------------

        med = np.median(R, axis=0)
        scale = 1.4826 * np.median(
            np.abs(R - med[None, :]),
            axis=0
        )
        scale = np.maximum(scale, 1e-12)

        # --------------------------------------------------------
        # Huber weights
        # --------------------------------------------------------

        t = np.abs(R) / scale[None, :]

        W = np.ones_like(R)
        mask = t > 1
        W[mask] = 1 / t[mask]
        W_total = W * meas_w[:, None]

        # --------------------------------------------------------
        # Solve each spectrum
        # --------------------------------------------------------

        ATA = np.einsum(
            "ma,mb,mn->nab",
            A,
            A,
            W_total,
            optimize=True,
        )
        ATY = np.einsum(
            "ma,mn,mn->na",
            A,
            W_total,
            Y,
            optimize=True,
        )

        ATA += reg * I

        coef_new = np.linalg.solve(
            ATA,
            ATY[..., None]
        )[..., 0].T

        # --------------------------------------------------------
        # Convergence
        # --------------------------------------------------------

        delta = np.max(np.abs(coef_new - coef))
        coef = coef_new

        if delta < tol:
            break

    return coef

def fit_harmonics(arr, angles, harmonics=(0,4,8), reg=1e-8, weights=None, robust=False):
    """
    Fit harmonic coefficients to angle-dependent data.
    arr: ndarray with angles on axis 0, remaining dims (e.g. scans, steps)
         shape (n_angles, N) or (n_angles, n_scans, n_steps)
    angles: 1D angles array length n_angles
    harmonics: iterable of ints
    reg: Tikhonov regularization scalar (applied to normal matrix)
    weights: optional per-angle weights (length n_angles) for weighted LS
    robust: if True, fall back to per-column Huber loss fitting (slower)
    Returns: coeffs array shape (n_coeffs, ...) (aligned to input remaining dims),
             and keys (list of column descriptors).
    """
    A, keys = build_design_matrix(angles, harmonics)
    n_angles, n_cols = A.shape

    if weights is None:
        w_vec = np.ones(n_angles)
        w_mat = w_vec[:, None]
    else:
        w_vec = np.asarray(weights)
    w_mat = w_vec[:, None]

    arr = np.asarray(arr)
    if arr.ndim == 1:
        Y = arr.reshape(n_angles, 1)
    else:
        Y = arr.reshape(n_angles, -1)

    # apply weights if given
    A_w = A * w_mat
    Y_w = Y * w_mat

    # Precompute regularized inverse (normal equations)
    ATA = A_w.T @ A_w  # (n_cols, n_cols)
    if reg is not None and reg > 0:
        ATA = ATA + reg * np.eye(ATA.shape[0])
    ATY = A_w.T @ Y_w  # (n_cols, n_pairs)
    try:
        coef = np.linalg.solve(ATA, ATY)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(ATA) @ ATY

    # If robust requested, optionally refine each column with Huber (slower)
    if robust:
        coef = robust_ridge_irls(
            A,
            Y,
            coef,
            reg=reg,
            weights=weights,
        )

    # reshape coef back to original trailing dims
    out_shape = arr.shape[1:] if arr.ndim > 1 else ()
    coef = coef.reshape((n_cols,) + out_shape)
    return coef, keys

def get_model(arr, angles, method='projection', harmonics=(0,4,8), reg=1e-8, weights=None, robust=False):
    """
    Compute model coefficients from angle-dependent data.
    - `method='projection'` uses trapezoidal projection
    - `method in ('lsq','lsq_robust')` uses linear least-squares fit to requested harmonics
    Returns coefficients array with ordering matching the requested harmonics' columns.
    """
    method = method.lower()
    if method == 'projection':
        coef = []
        for k in harmonics:
            if k == 0:
                integral = np.average(arr, axis=0)
                coef.append(integral)
            else:
                cos = np.cos(k * angles)
                sin = np.sin(k * angles)
                extra_axes = max(0, arr.ndim - 1)
                if extra_axes:
                    idx = (slice(None),) + (None,) * extra_axes
                    cos = cos[idx]
                    sin = sin[idx]
                integral_c = np.trapezoid(arr * cos, angles, axis=0) / np.trapezoid(cos**2, angles, axis=0)
                integral_s = np.trapezoid(arr * sin, angles, axis=0) / np.trapezoid(sin**2, angles, axis=0)
                coef.append(integral_c)
                coef.append(integral_s)
        return np.array(coef)
    elif method in ('lsq', 'lsq_robust'):
        do_robust = robust or (method == 'lsq_robust')
        coef, keys = fit_harmonics(arr, angles, harmonics=harmonics, reg=reg, weights=weights, robust=do_robust)
        # coef shape (n_keys, ...) return as-is
        return coef
    else:
        raise ValueError(f"Unknown method '{method}' for get_model")
