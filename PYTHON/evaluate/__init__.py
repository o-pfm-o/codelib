from . import fitting_engine
from . import math_functions

import numpy as np

def moving_average(x, N=1, kind='simple', axis=-1, weights=None):
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

import numpy as np
from typing import Tuple, Union


def calc_root_linear_interpolation(
    x1: float, x2: float, y1: float, y2: float
) -> float:
    """
    Calculate the x-coordinate where a line segment between two points crosses the x-axis.

    This function performs a linear interpolation between two points (x1, y1) and (x2, y2),
    estimating where the line crosses y = 0 (the root).

    Args:
        x1 (float): x-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y1 (float): y-coordinate of the first point.
        y2 (float): y-coordinate of the second point.

    Returns:
        float: Estimated x-coordinate of the root via linear interpolation.
    """
    # Linear interpolation formula for zero crossing
    return x1 - y1 * (x2 - x1) / (y2 - y1)


def calc_root_discrete(
    x: np.ndarray,
    y: np.ndarray,
    x_err: Union[float, np.ndarray] = 0.0,
    y_err: Union[float, np.ndarray] = 0.0,
) -> Tuple[float, float]:
    """
    Estimate the root (x-intercept) from discrete x-y data using linear interpolation.

    The function finds where y changes sign, indicating a zero-crossing, then interpolates
    between the two nearest data points. Optionally, it propagates uncertainties in both
    x and y values.

    Args:
        x (np.ndarray): Array of x-values.
        y (np.ndarray): Array of y-values.
        x_err (float or np.ndarray, optional): Error in x-values. Defaults to 0.0.
        y_err (float or np.ndarray, optional): Error in y-values. Defaults to 0.0.

    Returns:
        tuple: (x0, dx0)
            x0 (float): Estimated root from interpolation.
            dx0 (float): Uncertainty of the root (NaN if multiple or no roots found).
    """
    # Ensure x_err and y_err are arrays of same shape as x and y
    if isinstance(x_err, float):
        x_err = x_err * np.ones_like(x)
    if isinstance(y_err, float):
        y_err = y_err * np.ones_like(y)

    # Validate matching array sizes
    if not (x.size == y.size == x_err.size == y_err.size):  # type: ignore
        raise ValueError("Error: All input arrays must have the same size.")

    # Identify indices where y changes sign — potential roots
    zero_indices = np.where(np.diff(np.sign(y)))[0]

    # Ensure exactly one zero-crossing exists
    if zero_indices.size != 1:
        return float('NaN'), float('NaN')

    index = zero_indices[0]

    # Extract points and corresponding uncertainties
    x1, x2 = float(x[index]), float(x[index + 1])
    y1, y2 = float(y[index]), float(y[index + 1])
    dx1, dx2 = x_err[index], x_err[index + 1] # type: ignore
    dy1, dy2 = y_err[index], y_err[index + 1] # type: ignore

    # Compute interpolated root
    x0 = calc_root_linear_interpolation(x1, x2, y1, y2)

    # Propagate uncertainties to get root error
    dx0 = np.sqrt(
        ((-y1 * (x2 - x1) / (y2 - y1) ** 2 - (x2 - x1) / (y2 - y1)) * dy1) ** 2
        + (y1 * (x2 - x1) * dy2 / (y2 - y1) ** 2) ** 2
        + (dx1 * (1 + y1 / (y2 - y1))) ** 2
        + (dx2 * y2 / (y2 - y1)) ** 2
    )

    return x0, float(dx0)
