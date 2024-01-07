"""Affine feature map."""

from functools import cached_property
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import _check_feature_names_in

from neo_ls_svm._typing import FloatMatrix, FloatVector

F = TypeVar("F", np.float32, np.float64)


class AffineFeatureMap(BaseEstimator, TransformerMixin):
    """Affine feature map.

    Applies the transformation (x - shift) @ diag(1 / scale) @ A to the input row x for a given
    vector shift, vector scale, and matrix A.

    By setting append_features=True the transformed features are appended to the given features if
    the transformation matrix A is not None.
    """

    def __init__(
        self,
        *,
        scale: FloatVector[F],
        shift: FloatVector[F],
        A: FloatMatrix[F] | None = None,
        append_features: bool = False,
    ):
        self.scale = scale  # type: ignore[var-annotated]
        self.shift = shift  # type: ignore[var-annotated]
        self.A = A  # type: ignore[var-annotated]
        self.append_features = append_features

    def fit(
        self,
        X: FloatMatrix[F],
        y: FloatVector[F] | None = None,
        sample_weight: FloatVector[F] | None = None,
    ) -> "AffineFeatureMap":
        """Fit this transformer."""
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        scale = np.reshape(getattr(self, "scale_", self.scale), (-1, X.shape[1]))
        shift = np.reshape(getattr(self, "shift_", self.shift), (-1, X.shape[1]))
        A = getattr(self, "A_", self.A)
        assert scale.dtype == shift.dtype, "The scale and shift must have the same dtype"
        assert not np.any(scale == 0), "The scale may not be zero"
        assert np.all(np.isfinite(scale)), "The scale must be finite"
        assert np.all(np.isfinite(shift)), "The shift must be finite"
        assert (
            X.shape[1] == scale.shape[1]
        ), "The scale must be compatible with the number of features"
        assert (
            X.shape[1] == shift.shape[1]
        ), "The shift must be compatible with the number of features"
        if A is not None:
            assert (
                A.dtype == scale.dtype
            ), "The matrix A must have the same dtype as the scale and shift"
            assert (
                X.shape[1] == A.shape[0]
            ), "The matrix A must have rows equal to the number of features in X"
            assert np.all(np.isfinite(A)), "The matrix A must be finite"
        return self

    def transform(self, X: FloatMatrix[F]) -> FloatMatrix[F]:
        """Transform the given data with this transformer."""
        # Check the input.
        X = check_array(X)
        # Get the fitted version of the parameters, if they exist.
        scale = np.reshape(getattr(self, "scale_", self.scale), (-1, X.shape[1]))
        shift = np.reshape(getattr(self, "shift_", self.shift), (-1, X.shape[1]))
        A = getattr(self, "A_", self.A)
        # Transform the input.
        X_transformed: FloatMatrix[F] = (
            (X - shift) / scale
            if A is None
            else (
                X @ (A / scale.T) - shift @ (A / scale.T)  # Memory optimisation.
                if A.shape[1] < A.shape[0]
                else (X - shift) @ (A / scale.T)
            )
        )
        if self.append_features and A is not None:
            X_transformed = np.hstack((X, X_transformed))
        return X_transformed

    @cached_property
    def pseudo_inverse(self) -> FloatMatrix[F] | None:
        """The pseudo-inverse of this transform's matrix A."""
        return np.linalg.pinv(self.A) if self.A is not None else None

    def inverse_transform(self, X_transformed: FloatMatrix[F]) -> FloatMatrix[F]:
        """Approximately invert this transformation."""
        X: FloatMatrix[F] = check_array(X_transformed)
        scale = np.reshape(getattr(self, "scale_", self.scale), (-1, X.shape[1]))
        shift = np.reshape(getattr(self, "shift_", self.shift), (-1, X.shape[1]))
        A = getattr(self, "A_", self.A)
        if self.append_features and A is not None:
            X = X[:, : A.shape[0]]
        else:
            if A is not None:
                pinvA = cast(FloatMatrix[F], self.pseudo_inverse)
                X = X @ pinvA
            X = (X * scale + shift).astype(shift.dtype)
        return X

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        """Get output feature names for transformation."""
        A = getattr(self, "A_", self.A)
        input_features_array = cast(
            npt.NDArray[np.object_], _check_feature_names_in(self, input_features)
        )
        output_features: npt.NDArray[np.object_] = (
            input_features_array + "_shifted_scaled"
            if A is None
            else np.array(
                [f"{','.join(list(input_features_array))}_affine_map"] * A.shape[1],
                dtype=object,
            )
        )
        if self.append_features and A is not None:
            output_features = np.hstack((input_features_array, output_features))
        return output_features
