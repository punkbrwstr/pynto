import numpy as np

def expanding_mean(x, axis=None):
    nan_mask = np.isnan(x)
    cumsum = np.add.accumulate(np.where(nan_mask, 0, x))
    count = np.add.accumulate(np.where(nan_mask, 0, x / x))
    return np.where(nan_mask, np.nan, cumsum) / count

def expanding_var(x, axis=None):
    nan_mask = np.isnan(x)
    cumsum = np.add.accumulate(np.where(nan_mask, 0, x))
    cumsumOfSquares = np.add.accumulate(np.where(nan_mask, 0, x * x))
    count = np.add.accumulate(np.where(nan_mask, 0, x / x))
    return (cumsumOfSquares - cumsum * cumsum / count) / (count - 1)

def make_expanding(ufunc):
    def expanding(x, axis=None):
        mask = np.isnan(x)
        return np.where(mask, np.nan, ufunc.accumulate(np.where(mask,0,x)))
    return expanding

def ewma_vectorized_safe(data, alpha, row_size=None, dtype=None, order='C', out=None):
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float
    else:
        dtype = np.dtype(dtype)

    row_size = int(row_size) if row_size is not None else _get_max_row_size(alpha, dtype)

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
                       order='C', out=out_main_view)

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                        dtype=dtype, order='C', out=out[-trailing_n:])
    return out

def _get_max_row_size(alpha, dtype=float):
    assert 0. <= alpha < 1.
    epsilon = np.finfo(dtype).tiny
    return int(np.log(epsilon)/np.log(1-alpha)) + 1

def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None):
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out


def corrcoeff_1d(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(-1,keepdims=1)
    B_mB = B - B.mean(-1,keepdims=1)
    
    # Sum of squares
    ssA = np.einsum('i,i->',A_mA, A_mA)
    ssB = np.einsum('i,i->',B_mB, B_mB)
    
    # Finally get corr coeff
    return np.einsum('i,i->',A_mA,B_mB)/np.sqrt(ssA*ssB)

# https://stackoverflow.com/a/40085052/ @ Divakar
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

# https://stackoverflow.com/a/41703623/ @Divakar
def corr2_coeff_rowwise(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(-1,keepdims=1)
    B_mB = B - B.mean(-1,keepdims=1)

    # Sum of squares across rows
    ssA = np.einsum('ij,ij->i',A_mA, A_mA)
    ssB = np.einsum('ij,ij->i',B_mB, B_mB)

    # Finally get corr coeff
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)
def nancorrcoeff_1d(A,B):
    # Get combined mask
    comb_mask = ~(np.isnan(A) & ~np.isnan(B))
    count = comb_mask.sum()

    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - np.nansum(A * comb_mask,-1,keepdims=1)/count
    B_mB = B - np.nansum(B * comb_mask,-1,keepdims=1)/count

    # Replace NaNs with zeros, so that later summations could be computed    
    A_mA[~comb_mask] = 0
    B_mB[~comb_mask] = 0

    ssA = np.inner(A_mA,A_mA)
    ssB = np.inner(B_mB,B_mB)

    # Finally get corr coeff
    return np.inner(A_mA,B_mB)/np.sqrt(ssA*ssB)

def nancorrcoeff_rowwise(A,B):
    # Input : Two 2D arrays of same shapes (mxn). Output : One 1D array  (m,)
    # Get combined mask
    comb_mask = ~(np.isnan(A) & ~np.isnan(B))
    count = comb_mask.sum(axis=-1,keepdims=1)

    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - np.nansum(A * comb_mask,-1,keepdims=1)/count
    B_mB = B - np.nansum(B * comb_mask,-1,keepdims=1)/count

    # Replace NaNs with zeros, so that later summations could be computed    
    A_mA[~comb_mask] = 0
    B_mB[~comb_mask] = 0

    # Sum of squares across rows
    ssA = np.einsum('ij,ij->i',A_mA, A_mA)
    ssB = np.einsum('ij,ij->i',B_mB, B_mB)

    # Finally get corr coeff
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)

