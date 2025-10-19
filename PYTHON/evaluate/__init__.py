from . import fitting_engine
from . import math_functions

import numpy as np

def moving_average_general(x, N=1, kind='simple', axis=-1, weights=None):
    """
    Calculate moving averages for 1D or nD data along a specified axis.

    Parameters
    ----------
    x : array_like
        Input data, can be n-dimensional.
    N : int
        Window size.
    kind : str
        Type of moving average ('simple', 'exponential', 'weighted').
    axis : int
        Axis along which to compute the moving average.
    weights : array_like, optional
        Custom weights for the 'weighted' option.

    Returns
    -------
    out : ndarray
        Moving-averaged data with the same shape as `x`.
    """

    x = np.asanyarray(x)
    if N <= 1:
        return x.copy()

    if kind not in {'simple', 'exponential', 'weighted'}:
        raise ValueError("kind must be one of 'simple', 'exponential', 'weighted'")

    # Move the target axis to the end for easier handling
    x_moved = np.moveaxis(x, axis, -1)

    if kind == 'simple':
        kernel = np.ones(N) / N
    elif kind == 'weighted':
        if weights is None:
            raise ValueError("weights must be provided for 'weighted' moving average")
        weights = np.array(weights, dtype=float)
        if weights.size != N:
            raise ValueError("weights length must match N")
        kernel = weights / weights.sum()
    elif kind == 'exponential':
        alpha = 2 / (N + 1)
        result = np.empty_like(x_moved, dtype=float)
        result[..., 0] = x_moved[..., 0]
        for i in range(1, x_moved.shape[-1]):
            result[..., i] = alpha * x_moved[..., i] + (1 - alpha) * result[..., i - 1]
        return np.moveaxis(result, -1, axis)

    # Pad to preserve size
    pad_left, pad_right = N // 2, N - 1 - (N // 2)
    padded = np.pad(
        x_moved,
        [(0, 0)] * (x_moved.ndim - 1) + [(pad_left, pad_right)],
        mode='edge',
    )

    # Apply convolution along the last axis
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), -1, padded)

    # Restore original axis order
    return np.moveaxis(out, -1, axis)