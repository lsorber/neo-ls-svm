"""Power target transformer."""

from typing import Any, TypeVar

import numpy as np
from scipy.stats import yeojohnson, yeojohnson_normmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from neo_ls_svm._typing import FloatMatrix, FloatVector

F = TypeVar("F", np.float32, np.float64)


class PowerTargetTransformer(BaseEstimator, TransformerMixin):
    """Power target transformer.

    TODO
    """

    def __init__(self, max_epochs: int = 4):
        self.max_epochs = max_epochs

    def fit(
        self,
        X: FloatVector[F],
        y: FloatVector[F] | None = None,
        sample_weight: FloatVector[F] | None = None,
    ) -> "PowerTargetTransformer":
        """Fit this transformer."""
        # The target vector must be supplied as the feature matrix X.
        y_ = check_array(X, ensure_2d=False, dtype=(np.float64, np.float32)).ravel()
        # Learn a robust shift, scale, and power transform for the target.
        for _ in range(self.max_epochs):
            # Learn an optimal shift and scale for the target.
            l, self.y_shift_, u = np.quantile(y_, [0.05, 0.5, 0.95])  # noqa: E741
            self.y_scale_ = np.maximum(np.abs(l - self.y_shift_), np.abs(u - self.y_shift_))
            if self.y_scale_ <= np.finfo(y_.dtype).eps:
                self.y_scale = 1.0
            y_transformed = (y_ - self.y_shift_) / self.y_scale_
            # Learn a power transform for the target.
            self.lambda_ = yeojohnson_normmax(y_transformed)
            # Remove any outliers.
            y_transformed = yeojohnson(y_transformed, lmbda=self.lambda_)
            q1, q3 = np.quantile(y_transformed, [0.25, 0.75])
            iqr = q3 - q1
            outliers = (y_transformed < q1 - 1.5 * iqr) | (q3 + 1.5 * iqr < y_transformed)
            if not np.any(outliers) or np.all(outliers):
                break
            y_ = y_[~outliers]
        return self

    def transform(self, X: FloatMatrix[F]) -> FloatMatrix[F]:
        """Transform the given data with this transformer."""
        # The target vector must be supplied as the feature matrix X.
        y = check_array(X, ensure_2d=False, dtype=(np.float64, np.float32))
        y_transformed = yeojohnson((y - self.y_shift_) / self.y_scale_, lmbda=self.lambda_)
        y_transformed = y_transformed.astype(X.dtype)
        return y_transformed

    def inverse_transform(self, X_transformed: FloatMatrix[F]) -> FloatMatrix[F]:
        """Invert this transformation."""
        y_transformed = check_array(X_transformed, ensure_2d=False, dtype=(np.float64, np.float32))
        # Undo the power transform (adapted from sklearn.preprocessing.PowerTransformer).
        y = np.zeros_like(y_transformed)
        pos = y_transformed >= 0
        if abs(self.lambda_) < np.spacing(1.0):
            y[pos] = np.expm1(y_transformed[pos])
        else:
            y[pos] = np.expm1(
                np.log1p(
                    np.maximum(
                        y_transformed[pos] * self.lambda_, -1 + np.finfo(y_transformed.dtype).epsneg
                    )
                )
                / self.lambda_
            )
        if abs(self.lambda_ - 2) > np.spacing(1.0):
            y[~pos] = -np.expm1(
                np.log1p(
                    np.maximum(
                        -(2 - self.lambda_) * y_transformed[~pos],
                        -1 + np.finfo(y_transformed.dtype).epsneg,
                    )
                )
                / (2 - self.lambda_)
            )
        else:
            y[~pos] = -np.expm1(-y_transformed[~pos])
        # Undo the scale and shift.
        y = y * self.y_scale_ + self.y_shift_
        return y

    def _more_tags(self) -> dict[str, Any]:
        # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        return {"preserves_dtype": [np.float64, np.float32], "X_types": ["2darray", "1dlabels"]}
