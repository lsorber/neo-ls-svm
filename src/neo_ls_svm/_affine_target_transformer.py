"""Affine target transformer."""

from typing import Any, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from neo_ls_svm._typing import FloatMatrix, FloatVector

F = TypeVar("F", np.float32, np.float64)


class AffineTargetTransformer(BaseEstimator, TransformerMixin):
    """Affine target transformer.

    TODO
    """

    def fit(
        self,
        X: FloatVector[F],
        y: FloatVector[F] | None = None,
        sample_weight: FloatVector[F] | None = None,
    ) -> "AffineTargetTransformer":
        """Fit this transformer."""
        # The target vector must be supplied as the feature matrix X.
        y_ = check_array(X, ensure_2d=False, dtype=(np.float64, np.float32)).ravel()
        # Learn an optimal shift and scale for the target.
        l, self.y_shift_, u = np.quantile(y_, [0.05, 0.5, 0.95])  # noqa: E741
        self.y_scale_ = np.maximum(np.abs(l - self.y_shift_), np.abs(u - self.y_shift_))
        if self.y_scale_ <= np.finfo(y_.dtype).eps:
            self.y_scale = 1.0
        return self

    def transform(self, X: FloatMatrix[F]) -> FloatMatrix[F]:
        """Transform the given data with this transformer."""
        # The target vector must be supplied as the feature matrix X.
        y = check_array(X, ensure_2d=False, dtype=(np.float64, np.float32))
        y_transformed = (y - self.y_shift_) / self.y_scale_
        y_transformed = y_transformed.astype(X.dtype)
        return y_transformed

    def inverse_transform(self, X_transformed: FloatMatrix[F]) -> FloatMatrix[F]:
        """Invert this transformation."""
        y_transformed = check_array(X_transformed, ensure_2d=False, dtype=(np.float64, np.float32))
        # Undo the scale and shift.
        y = y_transformed * self.y_scale_ + self.y_shift_
        y = y.astype(y_transformed.dtype)
        return y

    def _more_tags(self) -> dict[str, Any]:
        # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        return {"preserves_dtype": [np.float64, np.float32], "X_types": ["2darray", "1dlabels"]}
