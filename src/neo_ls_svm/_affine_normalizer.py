"""Supervised affine normalizer."""

from typing import TypeVar, cast

import numpy as np
from sklearn.utils import check_consistent_length, check_X_y

from neo_ls_svm._affine_feature_map import AffineFeatureMap
from neo_ls_svm._quantizer import sample_bins_quantized_ecdf
from neo_ls_svm._typing import FloatMatrix, FloatVector
from neo_ls_svm._weighted_quantile import weighted_quantile

F = TypeVar("F", np.float32, np.float64)


class AffineNormalizer(AffineFeatureMap):
    """Supervised affine normalizer.

    Applies the transformation (x - shift) @ diag(1 / scale) to the input row x so that:

        1. shift centers the features x so that the class bins are optimally separated, and
        2. scale scales the shifted features x so that the difference between two samples from
           different class bins is the separability between those class bins.

    To compute the shift and scale, the samples are first quantized into class bins. Then, each
    class bin's median and standard deviation are computed.

    A feature's shift is determined as a weighted sum of shifts, where each shift is the optimal
    shift for a pair of class bins. The weights of these shifts are the geometric mean of the class
    bins' sample weight and the separability of that pair of class bins.

    The separability of a pair of class bins is defined as:

        np.abs(X_bins_μ[j] - X_bins_μ[i]) / (X_bins_σ[i] + X_bins_σ[j])

    where X_bins_μ[k] and X_bins_σ[k] are the k'th class bin's center and standard deviation,
    respectively.

    The scale is analogously determined as a weighted sum of scales, where each scale is the sum of
    a pair of class bin standard deviations. In this way, the difference between two features after
    transformation corresponds to the separability between their class bins.
    """

    def __init__(self, *, append_features: bool = False) -> None:
        self.shift = 0.0
        self.scale = 1.0
        self.A = None
        self.append_features = append_features

    def fit(
        self,
        X: FloatMatrix[F],
        y: FloatVector[F] | None = None,
        sample_weight: FloatVector[F] | None = None,
    ) -> "AffineFeatureMap":
        """Fit this transformer."""
        # Validate X and y.
        X, y = check_X_y(X, y, dtype=(np.float64, np.float32))
        y = np.ravel(np.asarray(y)).astype(X.dtype)
        # Use uniform sample weights if none are provided.
        sample_weight_ = cast(
            FloatVector[F],
            np.ones(y.shape) if sample_weight is None else np.ravel(np.asarray(sample_weight)),
        ).astype(y.dtype)
        check_consistent_length(y, sample_weight_)
        # Separate X into bins according to y.
        y_quantized = sample_bins_quantized_ecdf(y)
        bins = [y_quantized == i for i in range(np.min(y_quantized), np.max(y_quantized) + 1)]
        X_bins = [X[bin, :] for bin in bins]  # noqa: A001
        n_bins = [np.sum(sample_weight_[bin]) for bin in bins]  # noqa: A001
        s_bins = [sample_weight_[np.newaxis, bin] / np.sum(sample_weight_[bin]) for bin in bins]  # noqa: A001
        # Exit early if there is only one bin.
        self.shift_: FloatVector[F]
        self.scale_: FloatVector[F]
        if len(X_bins) <= 1:
            self.shift_ = np.zeros((1, X.shape[1]), dtype=X.dtype)
            self.scale_ = np.ones((1, X.shape[1]), dtype=X.dtype)
            super().fit(X, y, sample_weight_)
            return self
        # Compute an optimal shift and scale.
        X_bins_μ: list[FloatVector[F]] = [
            weighted_quantile(X_bin, s_bin.T, 0.5, axis=0)
            for X_bin, s_bin in zip(X_bins, s_bins, strict=False)
        ]
        X_bins_σ: list[FloatVector[F]] = [
            s_bin @ np.abs(X_bin - X_bin_μ)
            for X_bin, s_bin, X_bin_μ in zip(X_bins, s_bins, X_bins_μ, strict=False)
        ]
        sign = np.zeros((1, X.shape[1]), dtype=X.dtype)
        sum_w: FloatVector[F] = np.zeros((1, X.shape[1]), dtype=X.dtype)
        self.shift_ = np.zeros((1, X.shape[1]), dtype=X.dtype)
        self.scale_ = np.zeros((1, X.shape[1]), dtype=X.dtype)
        for i in range(len(X_bins_μ) - 1):
            for j in range(i + 1, len(X_bins_μ)):
                # Compute the distance between class bin centroids and the total variance in the two
                # class bins.
                diff_μ: FloatVector[F] = X_bins_μ[j] - X_bins_μ[i]
                sum_σ: FloatVector[F] = np.maximum(X_bins_σ[i] + X_bins_σ[j], np.finfo(X.dtype).eps)
                # Determine the weight of this pair of class bins as the geometric mean of their
                # sample weight and separability.
                separability = np.abs(diff_μ) / sum_σ
                w: float = np.sqrt(
                    (n_bins[i] + n_bins[j]) * (0.5 + separability)
                )  # Regularised GM.
                # For this pair of class bins, compute the optimal separation threshold.
                alpha = np.clip(X_bins_σ[i] / sum_σ, 1e-6, 1.0 - 1e-6)
                self.shift_ = self.shift_ + w * (X_bins_μ[i] + alpha * diff_μ)
                self.scale_ = self.scale_ + w * sum_σ
                sign += w * np.sign(diff_μ)
                sum_w += w
        sign /= sum_w
        self.shift_ = self.shift_ / sum_w  # type: ignore[assignment]
        self.scale_ = self.scale_ / sum_w  # type: ignore[assignment]
        self.scale_[np.sign(sign) < 0] = -self.scale_[np.sign(sign) < 0]
        # Validate the learned parameters.
        super().fit(X, y, sample_weight_)
        return self
