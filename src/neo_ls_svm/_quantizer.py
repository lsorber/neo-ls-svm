"""Quantisation of numerical features."""

from typing import Any, Literal, TypeVar, cast, overload

import numba
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin, check_array
from sklearn.utils.validation import _check_feature_names_in

from neo_ls_svm._typing import FloatVector, GenericVector, IntegerVector, NumberMatrix, NumberVector

F = TypeVar("F", np.float32, np.float64)
M = TypeVar("M", np.float32, np.float64, np.int32, np.int64, np.intp)
N = TypeVar("N", np.float32, np.float64, np.int32, np.int64, np.intp)


@numba.jit(nopython=True, nogil=True, parallel=False, fastmath=True, cache=True)
def _next_knot(
    x: FloatVector[F], y: IntegerVector[np.int64], knot: int, max_bin_error: int, max_bin_size: int
) -> tuple[int, int]:
    # Find next_knot which is the first index after knot for which the linear interpolation between
    # [x[knot], x[next_knot]) has an error larger than max_bin_error, or the number of samples in
    # [x[knot], x[next_knot]) is larger than max_bin_size.
    min_a, max_a = 0.0, np.inf
    for next_knot in range(knot + 1, len(x)):
        # The bin count includes all data points in [knot, next_knot).
        bin_count = int(y[next_knot - 1] - y[knot - 1] if knot > 0 else y[next_knot - 1])
        # Return next_knot if we have accumulated more data points than than the maximum bin size.
        if bin_count > max_bin_size:
            break
        # Continue if the current bin consists of only one knot.
        if next_knot == knot + 1:
            continue
        # Update the minimum and maximum tangents that would cause the error within the current bin
        # to exceed the max error.
        dx, dy = x[next_knot - 1] - x[knot], y[next_knot - 1] - y[knot]
        max_a = min(max_a, (dy + max_bin_error) / dx)
        min_a = max(min_a, (dy - max_bin_error) / dx)
        # If the tangent exceeds the minimum or maximum error, return the bin.
        a = dy / dx
        if not (min_a <= a <= max_a):
            break
    return next_knot, bin_count


@numba.jit(nopython=True, nogil=True, parallel=False, fastmath=True, cache=True)
def _prev_knot(
    x: FloatVector[F], y: IntegerVector[np.int64], knot: int, max_bin_error: int, max_bin_size: int
) -> tuple[int, int]:
    # Find prev_knot which is the first index before knot for which the linear interpolation between
    # [x[prev_knot], x[knot]) has an error larger than max_bin_error, or the number of samples in
    # [x[prev_knot], x[knot]) is larger than max_bin_size.
    min_a, max_a = 0.0, np.inf
    for prev_knot in range(knot - 1, -1, -1):
        # The bin count includes all data points in [prev_knot, knot).
        bin_count = int(y[knot - 1] - y[prev_knot - 1] if prev_knot > 0 else y[knot - 1])
        # Return next_knot if we have accumulated more data points than than the maximum bin size.
        if bin_count > max_bin_size:
            break
        # Continue if the current bin consists of only one knot.
        if knot == prev_knot + 1:
            continue
        # Update the minimum and maximum tangents that would cause the error within the current bin
        # to exceed the max error.
        dx, dy = x[knot - 1] - x[prev_knot], y[knot - 1] - y[prev_knot]
        max_a = min(max_a, (dy + max_bin_error) / dx)
        min_a = max(min_a, (dy - max_bin_error) / dx)
        # If the tangent exceeds the minimum or maximum error, return the bin.
        a = dy / dx
        if not (min_a <= a <= max_a):
            break
    return prev_knot, bin_count


@overload
def hist_quantized_ecdf(
    x: NumberVector[N],
    *,
    density: Literal[True],
    max_bin_error: float = ...,
    max_bin_size: float = ...,
) -> tuple[FloatVector[F], FloatVector[F]]:
    ...


@overload
def hist_quantized_ecdf(
    x: NumberVector[N],
    *,
    density: Literal[False],
    max_bin_error: float = ...,
    max_bin_size: float = ...,
) -> tuple[IntegerVector[np.int64], FloatVector[F]]:
    ...


def hist_quantized_ecdf(
    x: NumberVector[N],
    *,
    density: bool = False,
    max_bin_error: float = 0.0125,
    max_bin_size: float = 0.125,
    merge_bin_size: float = 0.025,
) -> tuple[FloatVector[F] | IntegerVector[np.int64], FloatVector[F]]:
    """Compute a vector's histogram by quantizing its empirical cumulative distribution function."""
    # Convert relative error and bin size to absolute error and bin size.
    max_bin_error = int(max_bin_error * len(x))
    max_bin_size = int(max_bin_size * len(x))
    merge_bin_size = int(merge_bin_size * len(x))
    # Compute the empirical cumulative distribution function of x.
    x, counts = np.unique(x, return_counts=True)  # Sorted unique values of x, with counts.
    # Improve output quality by starting with the largest x values to avoid ending up with a tiny
    # residual bin with only a few samples. This conditioning could be improved upon by making the
    # algorithm double-sided.
    y = np.cumsum(counts)
    # Quantize x, y optimally one bin at a time.
    x_, y_ = np.append(x, np.inf), np.append(y, np.iinfo(y.dtype).max)
    x_, y_ = np.insert(x_, 0, -np.inf), np.insert(y_, 0, 0)
    knot_left = 1
    knot_right = len(x_) - 1
    bin_edges_left = [x[0]]
    bin_edges_right = [x[-1]]
    hist_left: list[int] = []
    hist_right: list[int] = []
    while knot_left < knot_right:
        # Determine next not from the left, and previous knot from the right.
        knot_left_prev = knot_left
        knot_right_prev = knot_right
        knot_left, bin_count_left = _next_knot(x_, y_, knot_left, max_bin_error, max_bin_size)
        knot_right, bin_count_right = _prev_knot(x_, y_, knot_right, max_bin_error, max_bin_size)
        # Add bin counts.
        hist_left.append(bin_count_left)
        hist_right.insert(0, bin_count_right)
        # Add bin edges.
        bin_edges_left.append(
            (x_[knot_left] + x_[knot_left - 1]) / 2 if knot_left > 0 else x_[knot_left]
        )
        bin_edges_right.insert(
            0, (x_[knot_right] + x_[knot_right - 1]) / 2 if knot_right > 0 else x_[knot_right]
        )
        # Check if we should stop.
        if knot_left == knot_right:
            bin_edges = bin_edges_left + bin_edges_right[1:]
            hist = hist_left + hist_right
            break
        if knot_left > knot_right:
            hist = (
                hist_left[:-1]
                + [y[-1] - np.sum(hist_left[:-1]) - np.sum(hist_right[1:])]
                + hist_right[1:]
            )
            bin_edges = bin_edges_left[:-1] + bin_edges_right[1:]
            break
        if y_[knot_right - 1] - y_[knot_left - 1] <= merge_bin_size:
            knot_center_left = int(np.floor((knot_left + knot_right) / 2))
            knot_center_right = int(np.ceil((knot_left + knot_right) / 2))
            bin_edge_center = (x_[knot_center_left] + x_[knot_center_right]) / 2
            hist = (  # TODO: Cover this with tests. E.g. assert np.sum(hist) == y[-1].
                hist_left[:-1]
                + [y_[knot_center_left] - y_[knot_left_prev - 1]]
                + [y_[knot_right_prev - 1] - y_[knot_center_right - 1]]
                + hist_right[1:]
            )
            bin_edges = bin_edges_left[:-1] + [bin_edge_center] + bin_edges_right[1:]
            break
    # Convert counts to a density if requested.
    floating_dtype: npt.DTypeLike = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    hist_arr = (np.array(hist) / y[-1]).astype(floating_dtype) if density else np.array(hist)
    bin_edges_arr = np.array(bin_edges).astype(floating_dtype)
    return hist_arr, bin_edges_arr


class Quantizer(BaseEstimator, TransformerMixin):
    """Quantizing encoder for numerical features.

    Maps numerical features to [1, num_bins] by quantizing them into dynamically sized bins.
    """

    def __init__(
        self,
        *,
        max_bin_error: float = 0.0125,
        max_bin_size: float = 0.125,
        append_invfreq: bool = False,
        dtype: npt.DTypeLike = np.intp,
    ):
        self.max_bin_error = max_bin_error
        self.max_bin_size = max_bin_size
        self.append_invfreq = append_invfreq
        self.dtype = dtype
        if append_invfreq and not np.issubdtype(dtype, np.floating):
            self.dtype = np.float32

    def fit(self, X: NumberMatrix[N], y: Any = None) -> "Quantizer":
        """Fit this transformer."""
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.X_hist_: list[IntegerVector[np.int64]] = []
        self.X_bin_edges_ = []
        for j in range(X.shape[1]):
            # Compute a variable-width histogram by quantizing X[:, j]'s ECDF.
            Xj_hist, Xj_bin_edges = hist_quantized_ecdf(
                X[:, j],
                density=False,
                max_bin_error=self.max_bin_error,
                max_bin_size=self.max_bin_size,
            )
            self.X_hist_.append(Xj_hist)
            self.X_bin_edges_.append(Xj_bin_edges)
        return self

    def transform(self, X: NumberMatrix[N]) -> NumberMatrix[M]:
        """Transform the given data with this transformer."""
        X_transformed = np.empty(
            (X.shape[0], (1 + self.append_invfreq) * X.shape[1]), dtype=self.dtype
        )
        for j in range(X.shape[1]):
            Xj_bin_indices = np.clip(
                np.searchsorted(self.X_bin_edges_[j], X[:, j], side="right") - 1,
                0,
                len(self.X_bin_edges_[j]) - 2,
            )
            X_transformed[:, j] = Xj_bin_indices
            if self.append_invfreq:
                X_transformed[:, X.shape[1] + j] = (
                    1 / len(self.X_hist_[j]) / self.X_hist_[j][Xj_bin_indices]
                )
        return X_transformed

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        """Get output feature names for transformation."""
        input_features_array = cast(
            npt.NDArray[np.object_], _check_feature_names_in(self, input_features)
        )
        output_features: npt.NDArray[np.object_] = np.array(input_features_array) + "_quantized"
        if self.append_invfreq:
            output_features = np.hstack(
                (output_features, np.array(input_features_array) + "_invfreq")
            )
        return output_features


def sample_bins_quantized_ecdf(x: GenericVector, **kwargs: Any) -> IntegerVector[np.intp]:
    """Compute optimal sample bins of a vector by quantizing its ECDF."""
    x_unique, x = np.unique(x, return_inverse=True)
    if len(x_unique) <= np.ceil(np.sqrt(len(x))):
        return x
    quantizer = Quantizer(dtype=np.intp, **kwargs)
    sample_bins: IntegerVector[np.intp] = quantizer.fit_transform(x[:, np.newaxis]).ravel()
    return sample_bins


def sample_weights_quantized_ecdf(x: GenericVector, **kwargs: Any) -> FloatVector[F]:
    """Compute optimal sample weights of a vector by quantizing its ECDF."""
    dtype: npt.DTypeLike = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    x_unique, x, x_counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(x_unique) <= np.ceil(np.sqrt(len(x))):
        return x_counts[x] / np.sum(x_counts)
    quantizer = Quantizer(append_invfreq=True, dtype=dtype, **kwargs)
    sample_weights: FloatVector[F] = quantizer.fit_transform(x[:, np.newaxis])[:, 1]
    return sample_weights
