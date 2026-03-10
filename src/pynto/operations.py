import numpy as np
import bottleneck as bn


def rank(inputs: np.ndarray, out: np.ndarray) -> None:
    out[:] = inputs.argsort(axis=1).argsort(axis=1)


def zero_first_op(x: np.ndarray, out: np.ndarray) -> None:
    out[1:] = x[1:]
    out[0] = 0.0


def zero_to_na_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.equal(x, 0), np.nan, x)


def is_na_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = np.where(np.isnan(x), 1, 0)


def inc_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = x + 1


def dec_op(x: np.ndarray, out: np.ndarray) -> None:
    out[:] = x - 1


def expanding_mean(x: np.ndarray) -> np.ndarray:
    csum = np.cumsum(x, axis=0)
    N = x.shape[0]
    counts = np.arange(1, N + 1, dtype=float)
    shape = [1] * x.ndim
    shape[0] = N
    counts = counts.reshape(shape)
    return csum / counts


def expanding_var(x: np.ndarray) -> np.ndarray:
    cumsumOfSquares = np.add.accumulate(x * x)
    cumsum = np.add.accumulate(x)
    N = x.shape[0]
    counts = np.arange(1, N + 1, dtype=np.float64)
    shape = [1] * x.ndim
    shape[0] = N
    counts = counts.reshape(shape)
    out = np.full(x.shape, np.nan)
    np.divide(
        cumsumOfSquares - cumsum * cumsum / counts,
        counts - 1,
        out=out,
        where=counts - 1 > 0,
    )
    return out


def expanding_std(x: np.ndarray) -> np.ndarray:
    return np.sqrt(expanding_var(x))


def expanding_lag(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape, x[0])


def expanding_ret(x: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(x)
    first_row_idx = mask.argmax(axis=0)
    has_value = mask.any(axis=0)
    cols = np.arange(x.shape[1])
    first = np.where(has_value, x[first_row_idx, cols], np.nan)
    out = np.full(x.shape, np.nan)
    np.divide(x, first, out=out, where=~np.isnan(first))
    return out - 1


def expanding_diff(x: np.ndarray) -> np.ndarray:
    return x - x[0]


def rolling_diff(x: np.ndarray, window: int) -> np.ndarray:
    window -= 1
    return np.concat([np.full((window, x.shape[1]), np.nan), x[window:] - x[:-window]])


def rolling_lag(x: np.ndarray, window: int) -> np.ndarray:
    window -= 1
    return np.concat([np.full((window, x.shape[1]), np.nan), x[:-window]])


def rolling_ret(x: np.ndarray, window: int) -> np.ndarray:
    window -= 1
    return np.concat(
        [np.full((window, x.shape[1]), np.nan), x[window:] / x[:-window] - 1]
    )


def rolling_cov(x: np.ndarray, window: int) -> np.ndarray:
    means = bn.move_mean(x, window, axis=0)
    meanXY = bn.move_mean(np.multiply.reduce(x, axis=1), window)
    return meanXY - np.multiply.reduce(means, axis=1)


def rolling_cor(x: np.ndarray, window: int) -> np.ndarray:
    vars_ = bn.move_var(x, window, axis=0)
    return rolling_cov(x, window) / np.multiply.reduce(vars_, axis=1)


def rolling_ewma(data: np.ndarray, window: int) -> np.ndarray:
    alpha = 2 / (window + 1.0)
    scale = np.power(1 - alpha, np.arange(data.shape[0] + 1))
    adj_scale = (alpha * scale[-2]) / scale[:-1]
    if len(data.shape) == 2:
        adj_scale = adj_scale[:, None]
        scale = scale[:, None]
    offset = data[0] * scale[1:]
    return np.add.accumulate(data * adj_scale) / scale[-2::-1] + offset


def rolling_ewv(data: np.ndarray, window: int, bias_correct: bool = True) -> np.ndarray:
    """
    Exponentially-weighted variance using the same EWMA definition as rolling_ewma.
    If bias_correct=True, applies the standard unbiased correction for weighted variance.
    """
    mean = rolling_ewma(data, window)
    delta2 = (data - mean) ** 2

    # "Raw" EWM variance = EWMA of squared deviations
    var = rolling_ewma(delta2, window)

    if not bias_correct:
        return var

    alpha = 2 / (window + 1.0)
    n = data.shape[0]

    # For early points, only t+1 observations exist; after that, cap at window.
    m = np.minimum(np.arange(1, n + 1), window)  # 1..window

    # Precompute sums of exponentially decaying weights up to window
    j = np.arange(window)
    ws = (1 - alpha) ** j
    csum_w = np.cumsum(ws)
    csum_w2 = np.cumsum(ws**2)

    w_sum_t = csum_w[m - 1]
    w2_sum_t = csum_w2[m - 1]

    denom = (w_sum_t**2) - w2_sum_t

    # Unbiased correction factor; undefined when denom == 0 (e.g., m == 1)
    bias = np.divide(
        (w_sum_t**2),
        denom,
        out=np.full_like(w_sum_t, np.nan, dtype=float),
        where=denom > 0,
    )

    if data.ndim == 2:
        bias = bias[:, None]

    return var * bias


def rolling_ews(data: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(rolling_ewv(data, window))


def rolling_zsc(data: np.ndarray, window: int) -> np.ndarray:
    std = bn.move_std(data, window=window, axis=0, min_count=2)
    return np.divide(
        (data - bn.move_mean(data, window=window, axis=0, min_count=2)),
        std,
        out=None,
        where=std != 0,
    )
