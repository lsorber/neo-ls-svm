"""Weighted quantiles."""

from typing import Literal, TypeVar, overload

import numba
import numpy as np

from neo_ls_svm._typing import FloatMatrix, FloatTensor, FloatVector

F = TypeVar("F", np.float32, np.float64)


@numba.jit(nopython=True, nogil=True, parallel=True, fastmath=True, cache=True)
def _parallel_interp(q: FloatVector[F], p: FloatMatrix[F], a: FloatMatrix[F]) -> FloatMatrix[F]:
    out = np.empty((a.shape[0], len(q)), dtype=a.dtype)
    for i in numba.prange(a.shape[0]):
        out[i, :] = np.interp(q, p[i, :], a[i, :])
    return out


@overload
def weighted_quantile(
    a: FloatTensor[F], w: FloatTensor[F], q: float | FloatVector[F], axis: int
) -> FloatTensor[F]:
    ...


@overload
def weighted_quantile(
    a: FloatTensor[F], w: FloatTensor[F], q: float | FloatVector[F], axis: Literal[None]
) -> FloatVector[F]:
    ...


def weighted_quantile(
    a: FloatTensor[F],
    w: FloatTensor[F],
    q: float | FloatVector[F],
    axis: int | None = None,
) -> FloatTensor[F] | FloatVector[F]:
    """Compute the weighted q'th quantile of the data along the specified axis."""
    assert a.ndim == w.ndim, "Array and weights must have the same number of dimensions"
    assert axis is None or (0 <= axis < a.ndim), "Axis must be one of the array's dimensions"
    assert np.all(w >= 0), "Weights must be nonnegative"
    a, w = np.ascontiguousarray(a), np.ascontiguousarray(w)
    w = np.broadcast_to(w, a.shape)  # TODO: Short-circuit to np.quantile if weights are constant.
    q = np.ravel(np.asarray([q])).astype(a.dtype)
    if axis is not None:
        a, w = np.moveaxis(a, axis, -1), np.moveaxis(w, axis, -1)
        a_shape = a.shape
        a, w = np.reshape(a, [-1, a.shape[-1]]), np.reshape(w, [-1, w.shape[-1]])
        idx = np.argsort(a, axis=1)
        a, w = np.take_along_axis(a, idx, axis=1), np.take_along_axis(w, idx, axis=1)
        p = np.cumsum(w, axis=1)
        w_sum = p[:, [-1]].copy()
        p_lower = (p - w) / w_sum
        p_upper = p / w_sum
        tensor: FloatTensor[F] = (
            _parallel_interp(q, p_lower, a) + _parallel_interp(q, p_upper, a)
        ) / 2
        tensor = np.reshape(tensor, a_shape[:-1] + (len(q),))
        tensor = np.moveaxis(tensor, -1, axis)
        output = tensor
    else:
        a, w = np.ravel(a), np.ravel(w)
        idx = np.argsort(a)
        a, w = a[idx], w[idx]
        p = np.cumsum(w)
        # The definition below produces the desired result of 0.5 for the toy example a=(0, 1, 1),
        # w=(2, 1, 1), q=0.5, while the more standard `p = (p - 0.5 * w) / p[-1]` does not.
        p_lower = (p - w) / p[-1]
        p_upper = p / p[-1]
        vector: FloatVector[F] = (
            0.5 * np.interp(q, p_lower, a) + 0.5 * np.interp(q, p_upper, a)
        ).astype(a.dtype)
        output = vector
    return output
