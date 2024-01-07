"""Neo LS-SVM."""

from typing import Any, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh, lu_factor, lu_solve
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.utils.validation import check_consistent_length, check_X_y

from neo_ls_svm._affine_feature_map import AffineFeatureMap
from neo_ls_svm._affine_separator import AffineSeparator
from neo_ls_svm._feature_maps import (
    KernelApproximatingFeatureMap,
    OrthogonalRandomFourierFeatures,
)
from neo_ls_svm._typing import (
    ComplexMatrix,
    ComplexVector,
    FloatMatrix,
    FloatVector,
    GenericVector,
)

T = TypeVar("T", np.complex64, np.complex128)
F = TypeVar("F", np.float32, np.float64)


class NeoLSSVM(BaseEstimator):
    """Neo LS-SVM.

    A neo Least-Squares Support Vector Machine with:

      - [x] A next-generation regularisation term that penalises the complexity of the prediction
            surface, decision function, and maximises the margin.
      - [x] Large-scale support through state-of-the-art random feature maps.
      - [x] Optional automatic selection of primal or dual problem.
      - [x] Automatic optimal tuning of the regularisation hyperparameter γ that minimises the
            leave-one-out error, without having to refit the model.
      - [x] Automatic tuning of the kernel parameters σ, without having to refit the model.
      - [x] Automatic robust shift and scaling of the feature matrix and labels.
      - [x] Leave-one-out residuals, leverage, influence, and error as a free output after fitting,
            optimally clipped in classification.
      - [x] Isotonically calibrated class probabilities based on leave-one-out predictions.
      - [ ] Automatic robust fit by removing outliers.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        primal_feature_map: KernelApproximatingFeatureMap | None = None,
        dual_feature_map: AffineSeparator | None = None,
        dual: bool | None = False,
        max_epochs: int = 1,
        refit: bool = False,
        random_state: int | np.random.RandomState | None = 42,
        estimator_type: Literal["classifier", "regressor"] | None = None,
    ) -> None:
        self.primal_feature_map = primal_feature_map
        self.dual_feature_map = dual_feature_map
        self.dual = dual
        self.max_epochs = max_epochs
        self.refit = refit
        self.random_state = random_state
        self.estimator_type = estimator_type

    def _optimize_β̂_γ(
        self,
        C: FloatMatrix[F],
        φ: ComplexMatrix[T],
        y: FloatVector[F],
        s: FloatVector[F],
    ) -> tuple[ComplexVector[T], float, ComplexMatrix[T]]:
        """Find β̂ = argmin ||S(φ(X)β̂ - y)||² + γβ̂'Cβ̂ and γ that minimises the leave-one-out error.

        First, we solve min ||S(φ(X)β̂ - y)||² + γβ̂'Cβ̂ for β̂ as a function of γ::

          (γC + φ(X)'SSφ(X)) β̂ = φ(X)'S Sy
          (γI + C⁻¹φ(X)'SSφ(X)) β̂ = C⁻¹φ(X)'S Sy
          (yQQ⁻¹ + QΛQ⁻¹) β̂ = C⁻¹φ(X)'S Sy  where  φ(X)'SSφ(X)Q = CQΛ
          β̂ = Q(γI + Λ)⁻¹Q⁻¹ C⁻¹φ(X)'S Sy

        The entries of β̂ are rational polynomials of γ: Q diag([rᵢ(γ)]ᵢ) Q⁻¹ C⁻¹φ(X)'SSy. The
        weighted leave-one-out residuals eᵢ are given by [1]::

          eᵢ := sᵢ(φ(X)ᵢβ̂ - yᵢ) / (1 - hᵢ)
          hᵢ := sᵢφ(X)ᵢ (γC + φ(X)'SSφ(X))⁻¹ sᵢφ(X)ᵢ'  where  φ(X)ᵢ is the i'th row of φ(X)
              = sᵢφ(X)ᵢ Q(γI + Λ)⁻¹Q⁻¹C⁻¹    sᵢφ(X)ᵢ'

        The entries of hᵢ are also rational polynomials of γ: sᵢφ(X)ᵢQ diag([rᵢ(γ)]ᵢ) Q⁻¹C⁻¹φ(X)ᵢ'.

        We find the γ that optimises the weighted mean absolute leave-one-out error (WMAE) s'abs(e)
        by sampling sufficient γs and picking the best one.

        References
        ----------
        [1] https://robjhyndman.com/hyndsight/loocv-linear-models/
        """
        # Compute the GEVD φ(X)'SSφ(X)Q = CQΛ so that QΛQ⁻¹ = C⁻¹φ(X)'SSφ(X).
        Sφ = s[:, np.newaxis] * φ
        A = Sφ.conj().T @ Sφ
        A = (A + A.conj().T) / 2  # Ensure A is fully Hermitian.
        c = np.diag(C)
        C_is_diagonal = np.all(np.diag(c) == C)
        if C_is_diagonal:
            λ, Q = eigh((1 / c[:, np.newaxis]) * A)  # Scipy's eigh is faster for complex matrices.
            CQ_inv = Q.conj().T * (1 / c[np.newaxis, :])
        else:
            λ, Q = eigh(a=A, b=C)
            CQ_lu = lu_factor(C @ Q)
        # Compute the optimal parameters β̂ = Q(γI + Λ)⁻¹Q⁻¹C⁻¹φ(X)'SSy as a function of γ. We can
        # evaluate β̂(γ) as β̂ @ (1 / (γ + λ)) for a given γ.
        φSTSy = Sφ.conj().T @ (s * y)
        β̂ = (
            Q * (CQ_inv @ φSTSy)[np.newaxis, :]
            if C_is_diagonal
            else Q * lu_solve(CQ_lu, φSTSy)[np.newaxis, :]
        )
        # Compute part of the leave-one-out residual numerator sᵢ(yᵢ - φ(X)ᵢβ̂).
        φβ̂ = np.real(φ @ β̂)
        # Compute the leverage part hᵢ of the leave-one-out residual denominator 1 - hᵢ.
        h = (
            np.real(Sφ @ Q * (CQ_inv @ Sφ.conj().T).T)
            if C_is_diagonal
            else np.real(Sφ @ Q * lu_solve(CQ_lu, Sφ.conj().T).T)
        )
        # After np.real, arrays are _neither_ C nor F contiguous, which destroys performance.
        φβ̂ = np.ascontiguousarray(φβ̂)
        h = np.ascontiguousarray(h)
        # Evaluate the unweighted leave-one-out residuals for a set of γs with two matrix-matrix
        # products and pick the best solution.
        complexity_weights = np.logspace(np.log10(1e-6), np.log10(0.9), 1024, dtype=y.dtype)
        error_weights = 1 - complexity_weights
        self.γs_: FloatVector[F] = complexity_weights / error_weights
        rγ = 1 / (self.γs_[np.newaxis, :] + λ[:, np.newaxis])
        with np.errstate(divide="ignore", invalid="ignore"):
            unweighted_loo_residuals = (φβ̂ @ rγ - y[:, np.newaxis]) / (1 - h @ rγ)
            loo_residuals = self.y_scale_ * unweighted_loo_residuals
        # In the case of binary classification, clip overly positive and overly negative
        # predictions' residuals to 0 when the labels are positive and negative, respectively.
        if self._estimator_type == "classifier":
            loo_residuals[(y > 0)[:, np.newaxis] & (loo_residuals > 0)] = 0
            loo_residuals[(y < 0)[:, np.newaxis] & (loo_residuals < 0)] = 0
        # Select y that minimises the number of LOO misclassifications, the degree to which
        # LOO instances are misclassified, and the weighted absolute LOO error.
        self.loo_errors_γs_ = (s**2) @ np.abs(loo_residuals)
        optimum = np.argmin(
            (s**2) @ (np.abs(loo_residuals) >= 1)
            + (s**2) @ np.maximum(0, np.abs(loo_residuals) - 1)
            + self.loo_errors_γs_
            if self._estimator_type == "classifier"
            else self.loo_errors_γs_
        )
        # Store the leave-one-out residuals, leverage, influence, error, and score.
        self.loo_residuals_ = loo_residuals[:, optimum]
        self.loo_leverage_ = h @ rγ[:, optimum]
        dfbetas = (  # := β̂₍ᵢ₎ - β̂
            ((Q @ (rγ[:, optimum] * Q.conj().T)) @ Sφ.conj().T)
            * s[np.newaxis, :]
            * unweighted_loo_residuals[:, optimum][np.newaxis, :]
        )
        self.loo_error_ = self.loo_errors_γs_[optimum]
        if self._estimator_type == "classifier":
            # Compute the score as a weighted leave-one-out accuracy.
            self.loo_score_ = 1.0 - (s**2) @ (np.abs(self.loo_residuals_) > 1)
        elif self._estimator_type == "regressor":
            # Compute the score as a weighted leave-one-out R².
            ȳ = (s**2) @ y
            Sy, Sȳ = self.y_scale_ * s * y, self.y_scale_ * s * ȳ
            denom = (Sy - Sȳ) @ (Sy - Sȳ)
            numer = (s * self.loo_residuals_) @ (s * self.loo_residuals_)
            self.loo_score_ = 1.0 - numer / denom if denom > 0 else 1.0
        β̂, γ = β̂ @ rγ[:, optimum], self.γs_[optimum]
        # Resolve the linear system for better accuracy.
        if self.refit:
            β̂ = np.linalg.solve(γ * C + A, φSTSy)
        self.residuals_ = np.real(φ @ β̂) - y
        if self._estimator_type == "classifier":
            self.residuals_[(y > 0) & (self.residuals_ > 0)] = 0
            self.residuals_[(y < 0) & (self.residuals_ < 0)] = 0
        # TODO: Print warning if optimal γ is found at the edge.
        return β̂, γ, dfbetas

    def _optimize_α̂_γ(  # noqa: PLR0913
        self,
        C: FloatMatrix[F],
        X: FloatMatrix[F],
        y: FloatVector[F],
        s: FloatVector[F],
        gamma: float = 0.5,
    ) -> tuple[FloatVector[F], float]:
        """Find the dual solution to argmin ||S(φ(X)β̂ - y)||² + γβ̂'Cβ̂ that minimises the LOO error.

        The solution to argmin ||S(φ(X)β̂ - y)||² + γβ̂'Cβ̂ for β̂ as a function of α̂ is:

          (γC + φ(X)'SSφ(X))β̂ = φ(X)'S Sy
          (γ𝕀 + C⁻¹φ(X)'SSφ(X))β̂ = C⁻¹φ(X)'S Sy
          β̂ = (γ𝕀 + C⁻¹φ(X)'SSφ(X))⁻¹ C⁻¹φ(X)'S Sy
          β̂ = C⁻¹φ(X)'S (γ𝕀 + Sφ(X)C⁻¹φ(X)'S)⁻¹ Sy  with the identity  (γ𝕀 + AB)⁻¹A = A(γ𝕀 + BA)⁻¹
          β̂ = C⁻¹φ(X)' (γS⁻² + φ(X)C⁻¹φ(X)')⁻¹ y
          β̂ = C⁻¹φ(X)' α̂  where α̂ := (γS⁻² + φ(X)C⁻¹φ(X)')⁻¹ y

        Let's now make two modifications to the dual solution. First we'll regularise α̂ directly
        with γα̂'Cα̂ instead of regularising β̂, yielding:

          β̂ := φ(X)' α̂
          α̂ := (γS⁻¹CS⁻¹ + φ(X)φ(X)')⁻¹ y = (γS⁻¹CS⁻¹ + k(X, X))⁻¹ y
          ŷ(x) := φ(x)'β̂ = k(x, X)α̂

        Next we'll add a bias term to the prediction function in the dual space:

          α̂ := (γ[S⁻¹CS⁻¹, 0; 0, 1] + [k(X, X), 1; 1', 0])⁻¹ [y; 0] := (γD + K)⁻¹ [y; 0]
          ŷ(x) := [k(x, X) 1] α̂

        Now we can solve for α̂ as a function of γ:

          (γD + K) α̂ = [y; 0]
          (γ𝕀 + D⁻¹K) α̂ = D⁻¹ [y; 0]
          (γQQ⁻¹ + QΛQ⁻¹) α̂ = D⁻¹ [y; 0]  where  KQ = DQΛ
          α̂ = Q(γ𝕀 + Λ)⁻¹Q⁻¹D⁻¹ [y; 0]

        The entries of α̂ are rational polynomials of γ: Q diag([rᵢ(γ)]ᵢ) Q⁻¹D⁻¹ [y; 0]. The weighted
        leave-one-out residuals eᵢ can be derived by analogy to [1]::

          eᵢ := sᵢ(ŷ⁽⁻ᵏ⁾(xᵢ) - yᵢ)
             = -sᵢ[α̂ᵢ / (γD + K)⁻¹ᵢᵢ +
                   γ(S⁻¹CS⁻¹ - diag(S⁻¹CS⁻¹))ᵢ₌ (α̂ - (γD + K)⁻¹₌ᵢ / (γD + K)⁻¹ᵢᵢ)]
             = -sᵢ[α̂ᵢ / (Q(γ𝕀 + Λ)⁻¹Q⁻¹D⁻¹)ᵢᵢ + γ(S⁻¹CS⁻¹ - diag(S⁻¹CS⁻¹))ᵢ₌
                  (α̂ - (Q(γ𝕀 + Λ)⁻¹Q⁻¹D⁻¹)₌ᵢ / (Q(γ𝕀 + Λ)⁻¹Q⁻¹D⁻¹)ᵢᵢ)]

        The entries of eᵢ are also rational polynomials of γ.

        We find the γ that optimises the weighted mean absolute leave-one-out error (WMAE) s'abs(e)
        by sampling sufficient γs and picking the best one.

        References
        ----------
        [1] http://theoval.cmp.uea.ac.uk/publications/pdf/ijcnn2006a.pdf
        """
        # Construct D := [S⁻¹CS⁻¹, 0; 0, 1] and K := [k(X, X), 1; 1', 0].
        D = np.zeros((X.shape[0] + 1, X.shape[0] + 1), dtype=X.dtype)
        D[:-1, :-1] = (C / s[:, np.newaxis]) / s[np.newaxis, :]
        D[-1, -1] = np.mean(np.diag(D[:-1, :-1]))
        K = np.ones((X.shape[0] + 1, X.shape[0] + 1), dtype=X.dtype)
        K[:-1, :-1] = rbf_kernel(X, gamma=gamma)
        K[-1, -1] = 0
        # Compute the GEVD KQ = DQΛ so that QΛQ⁻¹ = D⁻¹K.
        d = np.diag(D)
        D_is_diagonal = np.all(np.diag(d) == D)
        if D_is_diagonal:
            λ, Q = eigh((1 / d[:, np.newaxis]) * K)  # Scipy's eigh is faster for complex matrices.
            DQ_inv = Q.conj().T * (1 / d[np.newaxis, :])
        else:
            λ, Q = eigh(a=K, b=D)
            DQ_lu = lu_factor(D @ Q)
            DQ_inv = lu_solve(DQ_lu, np.eye(D.shape[0], dtype=D.dtype))
        # Compute the optimal parameters â = Q(γI + Λ)⁻¹Q⁻¹D⁻¹ [y; 0] as a function of γ. We can
        # evaluate â(γ) as â @ (1 / (γ + λ)) for a given γ.
        y_zero = np.append(y, 0).astype(y.dtype)
        α̂ = (
            Q * (DQ_inv @ y_zero)[np.newaxis, :]
            if D_is_diagonal
            else Q * lu_solve(DQ_lu, y_zero)[np.newaxis, :]
        )
        # Compute hₖ := (Q(γI + Λ)⁻¹Q⁻¹D⁻¹)ₖₖ as a function of γ. We can evaluate h(γ) as
        # h @ (1 / (γ + λ)) for a given γ.
        h = Q * DQ_inv.T
        # Compute gₖ := (S⁻¹CS⁻¹ - diag(S⁻¹CS⁻¹))ₖ_.
        G = D[:-1, :-1].copy()
        np.fill_diagonal(G, 0)
        # Evaluate the unweighted leave-one-out residuals for a set of γs and pick the best one.
        complexity_weights = np.logspace(
            np.log10(1e-6), np.log10(0.9), 1024 if D_is_diagonal else 32, dtype=X.dtype
        )
        error_weights = 1 - complexity_weights
        self.γs_ = complexity_weights / error_weights
        rγ = 1 / (self.γs_[np.newaxis, :] + λ[:, np.newaxis])
        loo_residuals = (
            -self.y_scale_ * np.real(α̂[:-1, :] @ rγ) / np.real(h[:-1, :] @ rγ)
            if D_is_diagonal
            else -self.y_scale_
            * (
                np.real(α̂[:-1, :] @ rγ) / np.real(h[:-1, :] @ rγ)
                + self.γs_[np.newaxis, :] * (G @ np.real(α̂[:-1, :] @ rγ))
                - self.γs_[np.newaxis, :]
                / np.real(h[:-1, :] @ rγ)
                * np.einsum(
                    "ji,ir,rk,rj->jk",
                    G,
                    Q[:-1, :-1],
                    rγ[:-1, :],
                    DQ_inv[:-1, :-1],
                    optimize="optimal",
                )
            )
        )
        # In the case of binary classification, clip overly positive and overly negative
        # predictions' residuals to 0 when the labels are positive and negative, respectively.
        if self._estimator_type == "classifier":
            loo_residuals[(y > 0)[:, np.newaxis] & (loo_residuals > 0)] = 0
            loo_residuals[(y < 0)[:, np.newaxis] & (loo_residuals < 0)] = 0
        # Select γ that minimises the number of LOO misclassifications, the degree to which
        # LOO instances are misclassified, and the weighted absolute LOO error.
        self.loo_errors_γs_ = (s**2) @ np.abs(loo_residuals)
        optimum = np.argmin(
            (s**2) @ (np.abs(loo_residuals) >= 1)
            + (s**2) @ np.maximum(0, np.abs(loo_residuals) - 1)
            + self.loo_errors_γs_
            if self._estimator_type == "classifier"
            else self.loo_errors_γs_
        )
        # Store the leave-one-out residuals, leverage, error, and score.
        self.loo_residuals_ = loo_residuals[:, optimum]
        self.loo_leverage_ = 1.0 - np.real(h[:-1, :] @ rγ[:, optimum])
        self.loo_error_ = self.loo_errors_γs_[optimum]
        if self._estimator_type == "classifier":
            # Compute the score as a weighted leave-one-out accuracy.
            self.loo_score_ = 1.0 - (s**2) @ (np.abs(self.loo_residuals_) > 1)
        elif self._estimator_type == "regressor":
            # Compute the score as a weighted leave-one-out R².
            ȳ = (s**2) @ y
            Sy, Sȳ = self.y_scale_ * s * y, self.y_scale_ * s * ȳ
            denom = (Sy - Sȳ) @ (Sy - Sȳ)
            numer = (s * self.loo_residuals_) @ (s * self.loo_residuals_)
            self.loo_score_ = 1.0 - numer / denom if denom > 0 else 1.0
        α̂, γ = np.real(α̂ @ rγ[:, optimum]), self.γs_[optimum]
        # Resolve the linear system for better accuracy.
        if self.refit:
            α̂ = np.linalg.solve(γ * D + K, y_zero)
        self.residuals_ = (K[:-1, :-1] @ self.α̂_[:-1] + self.α̂_[-1]) - y
        if self._estimator_type == "classifier":
            self.residuals_[(y > 0) & (self.residuals_ > 0)] = 0
            self.residuals_[(y < 0) & (self.residuals_ < 0)] = 0
        # TODO: Print warning if optimal γ is found at the edge.
        return α̂, γ

    def fit(  # noqa: C901, PLR0912, PLR0915
        self, X: FloatMatrix[F], y: GenericVector, sample_weight: FloatVector[F] | None = None
    ) -> "NeoLSSVM":
        """Fit this predictor."""
        # Remove singleton dimensions from y and validate input.
        X, y = check_X_y(X, y, dtype=(np.float64, np.float32))
        y = np.ravel(np.asarray(y))
        self.n_features_in_ = X.shape[1]
        # Use uniform sample weights if none are provided.
        sample_weight_ = (
            np.ones(y.shape, X.dtype)
            if sample_weight is None
            else np.ravel(np.asarray(sample_weight))
        )
        check_consistent_length(y, sample_weight_)
        # Store the target's dtype for prediction.
        self.y_dtype_: npt.DTypeLike = y.dtype
        # Learn the type of task from the target. Set the `_estimator_type` and store the `classes_`
        # attribute if this is a classification task [1].
        # [1] https://scikit-learn.org/stable/glossary.html#term-classifiers
        y_: FloatVector[F]
        unique_y = np.unique(y)
        inferred_estimator_type = None
        if len(unique_y) == 2:  # noqa: PLR2004
            inferred_estimator_type = "classifier"
        elif (
            np.issubdtype(y.dtype, np.number)
            or np.issubdtype(y.dtype, np.datetime64)
            or np.issubdtype(y.dtype, np.timedelta64)
        ):
            inferred_estimator_type = "regressor"
        self._estimator_type: str | None = self.estimator_type or inferred_estimator_type
        if self._estimator_type == "classifier":
            self.classes_: GenericVector = unique_y
            negatives = y == self.classes_[0]
            y_ = np.ones(y.shape, dtype=X.dtype)
            y_[negatives] = -1
        elif self._estimator_type == "regressor":
            y_ = cast(npt.NDArray[np.floating[Any]], y)
        else:
            message = "Target type not supported"
            raise ValueError(message)
        # Fit robust shift and scale parameters for the target y.
        if self._estimator_type == "classifier":
            self.y_shift_: float = 0.0
            self.y_scale_: float = 1.0
        elif self._estimator_type == "regressor":
            l, self.y_shift_, u = np.quantile(y_, [0.05, 0.5, 0.95])  # noqa: E741
            self.y_scale_ = np.maximum(np.abs(l - self.y_shift_), np.abs(u - self.y_shift_))
        self.y_scale_ = 1.0 if self.y_scale_ <= np.finfo(X.dtype).eps else self.y_scale_
        y_ = ((y_ - self.y_shift_) / self.y_scale_).astype(X.dtype)
        # Initialise the primal and dual feature maps.
        self.primal_feature_map_ = clone(
            self.primal_feature_map or OrthogonalRandomFourierFeatures()
        )
        self.dual_feature_map_ = clone(self.dual_feature_map or AffineSeparator())
        # Determine whether we want to solve this in the primal or dual space.
        self.dual_ = X.shape[0] <= 768 if self.dual is None else self.dual  # noqa: PLR2004
        self.primal_ = not self.dual_
        # Learn an optimal distance metric for the primal or dual space and apply it to the feature
        # matrix X.
        if self.primal_:
            self.primal_feature_map_.fit(X, y_, sample_weight_)
            X = self.primal_feature_map_.transform(X)
        else:
            self.dual_feature_map_.fit(X, y_, sample_weight_)
            X = self.dual_feature_map_.transform(X)
        # Optimise the following sub-objectives for the weights β̂ and hyperparameter γ:
        # 1. Minimal mean squared error on training set: ||S(y - φ(X)β̂)||²
        # 2. Minimal complexity of the prediction surface: ∫||∇ₓφ(x)'β̂||²dx
        # 3. Maximal margin: ||β̂||².
        C: FloatMatrix[F]
        if self.primal_:
            C = self.primal_feature_map_.complexity_matrix.astype(X.dtype)
        else:
            gamma = 0.5
            C = np.sqrt(rbf_kernel(X, gamma=gamma)) * (
                1 - euclidean_distances(X, squared=True) * (gamma / X.shape[1])
            )
        # Combine sub-objectives (2) and (3) in a single regularisation matrix C.
        complexity_weight = 0.1
        margin_weight = 1 - complexity_weight
        C = complexity_weight * C + margin_weight * np.eye(C.shape[0], dtype=C.dtype)
        # Normalise C so that β̂'Cβ̂ is comparable in magnitude to the training error (1).
        C /= C.shape[0]
        # Fit a robust model by iteratively removing outlγing examples.
        for i in range(self.max_epochs):
            # Remove outlγing examples.
            if i > 0:
                # TODO: Add algorithm that reweights or removes outliers.
                keep = np.ones(y.shape, dtype=np.bool_)
                if np.all(keep):
                    break
                X, y_, sample_weight_ = X[keep, :], y_[keep], sample_weight_[keep]
                if self.dual_:
                    C = C[keep, :]
                    C = C[:, keep]
            # Normalise the sample weights.
            sample_weight_ = sample_weight_ / np.sum(sample_weight_)
            s = np.sqrt(sample_weight_)
            # Solve the primal or dual system for y that minimises the leave-one-out error.
            if self.primal_:
                self.β̂_, self.γ_, dfbetas = self._optimize_β̂_γ(C=C, φ=X, y=y_, s=s)  # type: ignore[type-var,var-annotated]
            else:
                self.α̂_, self.γ_ = self._optimize_α̂_γ(C=C, X=X, y=y_, s=s, gamma=gamma)
        # Calibrate probabilities with isotonic regression on the leave-one-out predictions.
        if self._estimator_type == "classifier":
            self.predict_proba_calibrator_ = IsotonicRegression(
                out_of_bounds="clip", y_min=0, y_max=1, increasing=True
            )
            loo_ŷ = self.loo_residuals_ + y_
            target = np.zeros_like(y_)
            target[y_ == np.max(y_)] = 1.0
            self.predict_proba_calibrator_.fit(loo_ŷ, target, sample_weight_)
        return self

    def decision_function(self, X: FloatMatrix[F]) -> FloatVector[F]:
        """Evaluate this predictor's decision function."""
        ŷ: FloatVector[F]
        if self.primal_:
            # Apply the feature map φ and predict as ŷ(x) := φ(x)'β̂.
            ŷ = np.real(
                cast(KernelApproximatingFeatureMap, self.primal_feature_map_).transform(X) @ self.β̂_
            )
        else:
            # Shift and scale X, then predict as ŷ(x) := [k(x, X) 1] â.
            X = cast(AffineFeatureMap, self.dual_feature_map_).transform(X)
            K = rbf_kernel(X, cast(FloatMatrix[F], self.X_), gamma=0.5)
            ŷ = K @ self.α̂_[:-1] + self.α̂_[-1]
        return ŷ

    def predict(self, X: FloatMatrix[F]) -> GenericVector:
        """Predict the output on a given dataset."""
        # Evaluate ŷ given the feature matrix X.
        ŷ_df = self.decision_function(X)
        if self._estimator_type == "classifier":
            # For binary classification, round to the nearest class label. When the decision
            # function is 0, we assign a negative class label [1].
            # [1] https://scikit-learn.org/stable/glossary.html#term-decision_function
            ŷ_df = np.sign(ŷ_df)
            ŷ_df[ŷ_df == 0] = -1
            # Remap to the original class labels.
            ŷ = self.classes_[((ŷ_df + 1) // 2).astype(np.intp)]
        elif self._estimator_type == "regressor":
            # Undo the label shift and scale.
            ŷ = ŷ_df.astype(np.float64) * self.y_scale_ + self.y_shift_
        # Map back to the training target dtype.
        ŷ = ŷ.astype(self.y_dtype_)
        return ŷ

    def predict_proba(self, X: FloatMatrix[F]) -> FloatMatrix[F]:
        """Predict the output probability (classification) or confidence interval (regression)."""
        if self._estimator_type == "classifier":
            ŷ_classification = self.decision_function(X)
            p = self.predict_proba_calibrator_.transform(ŷ_classification)
            P = np.hstack([1 - p[:, np.newaxis], p[:, np.newaxis]])
        else:
            # TODO: Replace point predictions with confidence interval.
            ŷ_regression = self.predict(X)
            P = np.hstack((ŷ_regression[:, np.newaxis], ŷ_regression[:, np.newaxis]))
        return P

    @property
    def loo_score(self) -> float:
        """Compute the leave-one-out score of this classifier or regressor."""
        return cast(float, self.loo_score_)

    def score(
        self, X: FloatMatrix[F], y: GenericVector, sample_weight: FloatVector[F] | None = None
    ) -> float:
        """Compute the accuracy or R² of this classifier or regressor."""
        ŷ = self.predict(X)
        score: float
        if self._estimator_type == "classifier":
            score = accuracy_score(y, ŷ, sample_weight=sample_weight)
        elif self._estimator_type == "regressor":
            # Cast to a numeric dtype in case the target is a datetime or timedelta.
            score = r2_score(
                y.astype(np.float64), ŷ.astype(np.float64), sample_weight=sample_weight
            )
        return score

    def _more_tags(self) -> dict[str, Any]:
        # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        return {"binary_only": True, "requires_y": True}
