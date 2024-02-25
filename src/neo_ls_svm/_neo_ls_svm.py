"""Neo LS-SVM."""

from typing import Any, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_factor, cho_solve, eigh, lu_factor, lu_solve
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor
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
    FloatTensor,
    FloatVector,
    GenericVector,
)

C = TypeVar("C", np.complex64, np.complex128)
F = TypeVar("F", np.float32, np.float64)


class NeoLSSVM(BaseEstimator):
    """Neo LS-SVM.

    A neo Least-Squares Support Vector Machine with:

        1. ⚡ Linear complexity in the number of training examples with Orthogonal Random Features.
        2. 🚀 Hyperparameter free: zero-cost optimization of the regularisation parameter γ and
             kernel parameter σ.
        3. 🏔️ Adds a new tertiary objective that minimizes the complexity of the prediction surface.
        4. 🎁 Returns the leave-one-out residuals and error for free after fitting.
        5. 🌀 Learns an affine transformation of the feature matrix to optimally separate the
             target's bins.
        6. 🪞 Can solve the LS-SVM both in the primal and dual space.
        7. 🌡️ Isotonically calibrated `predict_proba` based on the leave-one-out predictions.
        8. 🎲 Asymmetric conformal Bayesian confidence intervals for classification and regression.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        primal_feature_map: KernelApproximatingFeatureMap | Literal["auto"] = "auto",
        dual_feature_map: AffineSeparator | Literal["auto"] = "auto",
        dual: bool | Literal["auto"] = "auto",
        estimator_type: Literal["auto", "classifier", "regressor"] = "auto",
        random_state: int | np.random.RandomState | None = 42,
    ) -> None:
        self.primal_feature_map = primal_feature_map
        self.dual_feature_map = dual_feature_map
        self.dual = dual
        self.random_state = random_state
        self.estimator_type = estimator_type

    def _optimize_β̂_γ(
        self,
        φ: ComplexMatrix[C],
        y: FloatVector[F],
        s: FloatVector[F],
        C: FloatMatrix[F],
    ) -> tuple[ComplexVector[C], float]:
        """Find β̂ = argmin ||S(φ(X)β̂ - y)||² + γβ̂'Cβ̂ and γ that minimises the leave-one-out error.

        First, we solve min ||S(φ(X)β̂ - y)||² + γβ̂'Cβ̂ for β̂ as a function of γ::

            (γC + φ(X)'SSφ(X)) β̂ = φ(X)'S Sy
            (γ𝕀 + C⁻¹φ(X)'SSφ(X)) β̂ = C⁻¹φ(X)'S Sy
            (γQQ⁻¹ + QΛQ⁻¹) β̂ = C⁻¹φ(X)'S Sy  where  φ(X)'SSφ(X)Q = CQΛ
            β̂ = Q(γ𝕀 + Λ)⁻¹Q⁻¹ C⁻¹φ(X)'S Sy

        The entries of β̂ are rational polynomials of γ: Q diag([rᵢ(γ)]ᵢ) Q⁻¹ C⁻¹φ(X)'SSy. The
        unweighted leave-one-out residuals e⁽ˡᵒᵒ⁾ can be derived by analogy to [1]::

            eᵢ⁽ˡᵒᵒ⁾ := (φ(X)ᵢβ̂ - yᵢ) / (1 - hᵢ)  where  φ(X)ᵢ is the i'th row of φ(X)
            hᵢ := sᵢφ(X)ᵢ (γC + φ(X)'SSφ(X))⁻¹ sᵢφ(X)ᵢ'
                = sᵢφ(X)ᵢ Q(γ𝕀 + Λ)⁻¹Q⁻¹C⁻¹    sᵢφ(X)ᵢ'

        The entries of hᵢ are also rational polynomials of γ: sᵢφ(X)ᵢQ diag([rᵢ(γ)]ᵢ) Q⁻¹C⁻¹φ(X)ᵢ'.

        We find the γ that optimises the weighted mean absolute leave-one-out error s'|e⁽ˡᵒᵒ⁾| by
        sampling sufficient γs and picking the best one.

        References
        ----------
        [1] https://robjhyndman.com/hyndsight/loocv-linear-models/
        """
        # Normalise the sample weights.
        s = s / np.sum(s)
        # Compute the GEVD φ(X)'SSφ(X)Q = CQΛ so that QΛQ⁻¹ = C⁻¹φ(X)'SSφ(X).
        Sφ = s[:, np.newaxis] * φ
        A = Sφ.conj().T @ Sφ
        A = (A + A.conj().T) / 2  # Ensure A is fully Hermitian.
        c = np.diag(C)
        C_is_diagonal = np.all(np.diag(c) == C)
        C = C / np.mean(np.abs(c)) / φ.size  # Normalise C.
        c = c / np.mean(np.abs(c)) / φ.size  # Normalise c.
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
        self.γs_: FloatVector[F] = np.logspace(np.log10(1e-6), np.log10(20), 1024, dtype=y.dtype)
        rγ = 1 / (self.γs_[np.newaxis, :] + λ[:, np.newaxis])
        with np.errstate(divide="ignore", invalid="ignore"):
            loo_residuals = (φβ̂ @ rγ - y[:, np.newaxis]) / (1 - h @ rγ)
            ŷ_loo = y[:, np.newaxis] + loo_residuals
        # In the case of binary classification, clip overly positive and overly negative
        # predictions' residuals to 0 when the labels are positive and negative, respectively.
        if self._estimator_type == "classifier":
            loo_residuals[(y > 0)[:, np.newaxis] & (loo_residuals > 0)] = 0
            loo_residuals[(y < 0)[:, np.newaxis] & (loo_residuals < 0)] = 0
        # Select γ that minimises the number of LOO misclassifications, the degree to which
        # LOO instances are misclassified, and the weighted absolute LOO error.
        self.loo_errors_γs_ = s @ np.abs(loo_residuals)
        optimum = np.argmin(
            s @ (np.abs(loo_residuals) >= 1)
            + s @ np.maximum(0, np.abs(loo_residuals) - 1)
            + self.loo_errors_γs_
            if self._estimator_type == "classifier"
            else self.loo_errors_γs_
        )
        # Store the leave-one-out residuals, leverage, error, and score.
        self.loo_residuals_ = loo_residuals[:, optimum]
        self.loo_ŷ_ = y + self.loo_residuals_
        self.loo_leverage_ = h @ rγ[:, optimum]
        self.loo_error_ = self.loo_errors_γs_[optimum]
        if self._estimator_type == "classifier":
            self.loo_score_ = accuracy_score(y, np.sign(ŷ_loo[:, optimum]), sample_weight=s)
        elif self._estimator_type == "regressor":
            self.loo_score_ = r2_score(y, ŷ_loo[:, optimum], sample_weight=s)
        β̂, γ = β̂ @ rγ[:, optimum], self.γs_[optimum]
        # Resolve the linear system for better accuracy.
        self.L_ = cho_factor(γ * C + A)
        β̂ = cho_solve(self.L_, φSTSy)
        self.residuals_ = np.real(φ @ β̂) - y
        if self._estimator_type == "classifier":
            self.residuals_[(y > 0) & (self.residuals_ > 0)] = 0
            self.residuals_[(y < 0) & (self.residuals_ < 0)] = 0
        # Compute the leave-one-out nonconformity with the Sherman-Morrison formula.
        σ2 = np.real(np.sum(φ * cho_solve(self.L_, φ.conj().T).T, axis=1))
        σ2 = np.ascontiguousarray(σ2)
        loo_σ2 = σ2 + (s * σ2) ** 2 / (1 - self.loo_leverage_)
        self.loo_nonconformity_ = np.sqrt(loo_σ2)
        # TODO: Print warning if optimal γ is found at the edge.
        return β̂, γ

    def _optimize_α̂_γ(
        self,
        X: FloatMatrix[F],
        y: FloatVector[F],
        s: FloatVector[F],
        ρ: float = 1.0,
    ) -> tuple[FloatVector[F], float]:
        """Find the dual solution to argmin ℒ(e,β̂,b,α̂).

        The Lagrangian is defined as::

            ℒ(e,β̂,b,α̂) := 1/(2γρ) e'S²e + 1/2 (β̂'β̂ + b²) + (1-ρ)/(2ρ) α̂'Cα̂ - α̂'(φ(X)β̂ + b - y - e)

        where γ determines the weight of the regularisation terms β̂'β̂ and α̂'Cα̂, which maximise the
        margin and minimise the complexity of the prediction surface, respectively, and ρ determines
        the trade-off between these two regularisation terms. The residuals e are defined as
        e := φ(X)β̂ + b - y.

        There are two differences w.r.t. the classic LS-SVM formulation: we regularise b in addition
        to β̂, and we add a regularisation term for the complexity of the prediction surface of the
        form α̂'Cα̂. Furthermore, we assume that S = diag(s) and that C is symmetric.

        Setting the gradient of the Lagrangian to zero yields::

            ∂ℒ/∂e = 1/(γρ) S²e + α̂ = 0 => e = -γρ S⁻²α̂
            ∂ℒ/∂β̂ = β̂ - φ(X)'α̂ = 0 => β̂ = φ(X)'α̂
            ∂ℒ/∂b = b - 1'α̂ = 0 => b = 1'α̂
            ∂ℒ/∂α̂ = (1-ρ)/ρ Cα̂ - φ(X)β̂ - b + y + e = 0 => [φ(X)φ(X)' + 11' - (1-ρ)/ρ C + γρS⁻²]α̂ = y

        Let K := φ(X)φ(X)' + 11' - (1-ρ)/ρ C, then we can solve for α̂(γ)::

            (γρS⁻² + K) α̂ = y  and  ŷ(x) := k(x, X)α̂ + b
            S⁻¹(γρ𝕀 + SKS)S⁻¹ α̂ = y
            S⁻¹(γρQQ⁻¹ + QΛQ⁻¹)S⁻¹ α̂ = y  where  SKS Q = QΛ
            α̂ = SQ(γρ𝕀 + Λ)⁻¹Q⁻¹S y

        The entries of α̂ are rational polynomials of γ: Q diag([rᵢ(γ)]ᵢ) Q⁻¹D⁻¹ y.

        Next, we derive the unweighted leave-one-out residuals e⁽ˡᵒᵒ⁾ by analogy to [1]. First, we
        define F := φ(X)φ(X)' + 11', G := -(1-ρ)/ρ C + γρS⁻², and H := (F + G)⁻¹ so that::

            (F + G) α̂ = y
            [f₁₁+g₁₁ f₁'+g₁'; f₁+g₁ F₁+G₁] α̂ = y
            α̂ = Hy = [h₁₁ h₁'; h₁ H₁] y
            h₁₁ := 1/(f₁₁+g₁₁ - (f₁'+g₁')(F₁+G₁)⁻¹(f₁+g₁))
            h₁  := -h₁₁(F₁+G₁)⁻¹(f₁+g₁)
            ŷ₁⁽⁻¹⁾ := f₁' α̂⁽⁻¹⁾
                   = f₁' (F₁+G₁)⁻¹ y⁽⁻¹⁾
                   = f₁' (F₁+G₁)⁻¹ [f₁+g₁ F₁+G₁] α̂
                   = f₁'[-h₁/h₁₁ 𝕀] α̂
            y₁ = [f₁₁+g₁₁ f₁'+g₁'] α̂
            e₁⁽ˡᵒᵒ⁾ := ŷ₁⁽⁻¹⁾ - y₁ = f₁'[-h₁/h₁₁ 𝕀] α̂ - y₁

        We find the γ that optimises the weighted mean absolute leave-one-out error s'|e⁽ˡᵒᵒ⁾| by
        sampling sufficient γs and picking the best one.

        References
        ----------
        [1] http://theoval.cmp.uea.ac.uk/publications/pdf/ijcnn2006a.pdf
        """
        # Normalise the sample weights.
        s = s / np.sum(s)
        sn = s / np.median(np.abs(s))
        # Construct the regularisation matrix C that penalises the complexity of the prediction
        # surface. TODO: Document the derivation of this term.
        gamma = 0.5
        C = np.sqrt(rbf_kernel(X, gamma=gamma)) * (
            1 - euclidean_distances(X, squared=True) * (gamma / X.shape[1])
        )
        # Construct F := φ(X)φ(X)' + 11'.
        F = rbf_kernel(X, gamma=0.5) + np.ones(X.shape[0], dtype=X.dtype)
        # Construct D⁻¹K := 1/ρ S² [φ(X)φ(X)' + 11' - (1-ρ)/ρ C].
        K = F - (1 - ρ) / ρ * C
        # Compute the EVD SKS Q = QΛ.
        λ, Q = np.linalg.eigh(sn[:, np.newaxis] * K * sn[np.newaxis, :])
        # Compute the optimal parameters â = SQ(γρI + Λ)⁻¹Q⁻¹S y as a function of γ. We can evaluate
        # â(γ) as α̂ @ (1 / (γρ + λ)) for a given γ.
        α̂ = (sn[:, np.newaxis] * Q) * (Q.conj().T @ (sn * y))[np.newaxis, :]
        # Evaluate the unweighted leave-one-out residuals for a set of γs and pick the best one.
        self.γs_ = np.logspace(np.log10(1e-6), np.log10(20), 128, dtype=X.dtype)
        # Compute the leave-one-out predictions ŷ₁⁽⁻¹⁾ = f₁'[-h₁/h₁₁ 𝕀] α̂.
        H_loo = np.einsum(  # Compute H := SQ(γρI + Λ)⁻¹Q⁻¹S as a function of γ.
            "ij,gj,jk->igk",
            sn[:, np.newaxis] * Q,
            1 / (self.γs_[:, np.newaxis] * ρ + λ[np.newaxis, :]),
            Q.conj().T * sn[np.newaxis, :],
            optimize="optimal",
        )
        for g in range(H_loo.shape[1]):
            h = np.diag(H_loo[:, g, :]).copy()
            h[h == 0] = np.finfo(X.dtype).eps  # Avoid division by zero.
            H_loo[:, g, :] = H_loo[:, g, :] / -h[:, np.newaxis]
        F_loo = F.copy()
        np.fill_diagonal(F_loo, 0)
        α̂_loo = α̂ @ (1 / (self.γs_[np.newaxis, :] * ρ + λ[:, np.newaxis]))
        ŷ_loo = np.sum(F_loo[:, np.newaxis, :] * H_loo, axis=2) * α̂_loo + F_loo @ α̂_loo
        loo_residuals = ŷ_loo - y[:, np.newaxis]
        # In the case of binary classification, clip overly positive and overly negative
        # predictions' residuals to 0 when the labels are positive and negative, respectively.
        if self._estimator_type == "classifier":
            loo_residuals[(y > 0)[:, np.newaxis] & (loo_residuals > 0)] = 0
            loo_residuals[(y < 0)[:, np.newaxis] & (loo_residuals < 0)] = 0
        # Select γ that minimises the number of LOO misclassifications, the degree to which
        # LOO instances are misclassified, and the weighted absolute LOO error.
        self.loo_errors_γs_ = s @ np.abs(loo_residuals)
        optimum = np.argmin(
            s @ (np.abs(loo_residuals) >= 1)
            + s @ np.maximum(0, np.abs(loo_residuals) - 1)
            + self.loo_errors_γs_
            if self._estimator_type == "classifier"
            else self.loo_errors_γs_
        )
        # Store the leave-one-out residuals, leverage, error, and score.
        self.loo_residuals_ = loo_residuals[:, optimum]
        self.loo_ŷ_ = y + self.loo_residuals_
        self.loo_error_ = self.loo_errors_γs_[optimum]
        if self._estimator_type == "classifier":
            self.loo_score_ = accuracy_score(y, np.sign(ŷ_loo[:, optimum]), sample_weight=s)
        elif self._estimator_type == "regressor":
            self.loo_score_ = r2_score(y, ŷ_loo[:, optimum], sample_weight=s)
        α̂, γ = α̂_loo[:, optimum], self.γs_[optimum]
        # Resolve the linear system for better accuracy.
        self.L_ = cho_factor(γ * ρ * np.diag(sn**-2) + K)
        α̂ = cho_solve(self.L_, y)
        self.residuals_ = F @ α̂ - y
        if self._estimator_type == "classifier":
            self.residuals_[(y > 0) & (self.residuals_ > 0)] = 0
            self.residuals_[(y < 0) & (self.residuals_ < 0)] = 0
        # Compute the nonconformity. TODO: Apply a leave-one-out correction.
        K = rbf_kernel(X, X, gamma=0.5)
        σ2 = 1.0 - np.sum(K * cho_solve(self.L_, K.T).T, axis=1)
        self.loo_nonconformity_ = np.sqrt(σ2)
        # TODO: Print warning if optimal γ is found at the edge.
        return α̂, γ

    def fit(
        self, X: FloatMatrix[F], y: GenericVector, sample_weight: FloatVector[F] | None = None
    ) -> "NeoLSSVM":
        """Fit this predictor."""
        # Remove singleton dimensions from y and validate input.
        X, y = check_X_y(X, y, dtype=(np.float64, np.float32))
        y = np.ravel(np.asarray(y))
        self.n_features_in_ = X.shape[1]
        # Store the target's dtype for prediction.
        self.y_dtype_: npt.DTypeLike = y.dtype
        # Use uniform sample weights if none are provided.
        sample_weight_ = (
            np.ones(y.shape, X.dtype)
            if sample_weight is None
            else np.ravel(np.asarray(sample_weight)).astype(X.dtype)
        )
        check_consistent_length(y, sample_weight_)
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
        self._estimator_type: str | None = (
            inferred_estimator_type if self.estimator_type == "auto" else self.estimator_type
        )
        if self._estimator_type == "classifier":
            self.classes_: GenericVector = unique_y
            negatives = y == self.classes_[0]
            y_ = np.ones(y.shape, dtype=X.dtype)
            y_[negatives] = -1
        elif self._estimator_type == "regressor":
            y_ = y.astype(X.dtype)
        else:
            message = "Target type not supported"
            raise ValueError(message)
        # Determine whether we want to solve this in the primal or dual space.
        self.dual_ = X.shape[0] <= 1024 if self.dual == "auto" else self.dual  # noqa: PLR2004
        self.primal_ = not self.dual_
        # Learn an optimal distance metric for the primal or dual space and apply it to the feature
        # matrix X.
        if self.primal_:
            self.primal_feature_map_ = clone(
                OrthogonalRandomFourierFeatures()
                if self.primal_feature_map == "auto"
                else self.primal_feature_map
            )
            self.primal_feature_map_.fit(X, y_, sample_weight_)
            φ = self.primal_feature_map_.transform(X)
        else:
            nz_weight = sample_weight_ > 0
            X, y_, sample_weight_ = X[nz_weight], y_[nz_weight], sample_weight_[nz_weight]
            self.dual_feature_map_ = clone(
                AffineSeparator() if self.dual_feature_map == "auto" else self.dual_feature_map
            )
            self.dual_feature_map_.fit(X, y_, sample_weight_)
            self.X_ = self.dual_feature_map_.transform(X)
        # Solve the primal or dual system. We optimise the following sub-objectives for the weights
        # β̂ and hyperparameter γ:
        #   1. Minimal mean squared error on training set: ||S(y - φ(X)β̂)||²
        #   2. Maximal margin: ||β̂||².
        #   3. Minimal complexity of the prediction surface: ∫||∇ₓφ(x)'β̂||²dx
        if self.primal_:
            C = self.primal_feature_map_.complexity_matrix.astype(φ.dtype)
            self.β̂_, self.γ_ = self._optimize_β̂_γ(φ=φ, y=y_, s=sample_weight_, C=C)
        else:
            self.α̂_, self.γ_ = self._optimize_α̂_γ(X=self.X_, y=y_, s=sample_weight_)
        # Calibrate probabilities with isotonic regression on the leave-one-out predictions.
        if self._estimator_type == "classifier":
            self.predict_proba_calibrator_ = IsotonicRegression(
                out_of_bounds="clip", y_min=0, y_max=1, increasing=True
            )
            target = np.zeros_like(y_)
            target[y_ == np.max(y_)] = 1.0
            self.predict_proba_calibrator_.fit(self.loo_ŷ_, target, sample_weight_)
        # Lazily fit conformal predictors as quantile regression models that predict the lower and
        # upper bounds of the (relative) leave-one-out residuals.
        self.conformal_regressors_: dict[str, dict[float, QuantileRegressor]] = {
            "Δ⁺": {},
            "Δ⁻": {},
            "Δ⁺/ŷ": {},
            "Δ⁻/ŷ": {},
        }
        return self

    def nonconformity_measure(self, X: FloatMatrix[F]) -> FloatVector[F]:
        """Compute the nonconformity of a set of examples."""
        # Estimate the nonconformity as the variance of this model's Gaussian Process.
        σ2: FloatVector[F]
        if self.primal_:
            # If β̂ := (LL')⁻¹ y* and cov(y*) := LL', then cov(β̂) = cov((LL')⁻¹ y*) = (LL')⁻¹
            # assuming 𝔼(β̂) = 0. It follows that cov(ŷ(x)) = cov(φ(x)'β̂) = φ(x)'(LL')⁻¹φ(x).
            φH = cast(KernelApproximatingFeatureMap, self.primal_feature_map_).transform(X)
            σ2 = np.real(np.sum(φH * cho_solve(self.L_, φH.conj().T).T, axis=1))
            σ2 = np.ascontiguousarray(σ2)
        else:
            # Compute the cov(ŷ(x)) as K(x, x) − K(x, X) (LL')⁻¹ K(X, x). TODO: Document derivation.
            X = cast(AffineFeatureMap, self.dual_feature_map_).transform(X)
            K = rbf_kernel(X, self.X_, gamma=0.5)
            σ2 = 1.0 - np.sum(K * cho_solve(self.L_, K.T).T, axis=1)
        # Convert the variance to a standard deviation.
        σ = np.sqrt(σ2)
        return σ

    def predict_confidence_interval(
        self, X: FloatMatrix[F], *, confidence_level: float = 0.95
    ) -> FloatMatrix[F] | FloatTensor[F]:
        # Determine the quantiles at the edge of the confidence interval.
        quantile = 1 - (1 - confidence_level) / 2
        # Lazily fit any missing conformal regressors.
        # TODO: Perhaps exclude samples that were used in the feature map.
        # TODO: Perhaps enforce a positive slope for the nonconformity measure.
        for target_type in ("Δ⁺", "Δ⁻", "Δ⁺/ŷ", "Δ⁻/ŷ"):
            quantile_regressors = self.conformal_regressors_[target_type]
            if quantile not in quantile_regressors:
                sgn = (self.loo_residuals_ > 0) if "⁺" in target_type else (self.loo_residuals_ < 0)
                eps = np.finfo(self.loo_ŷ_.dtype).eps
                X_qr = np.hstack(
                    [
                        self.loo_nonconformity_[sgn, np.newaxis],
                        self.loo_ŷ_[sgn, np.newaxis],
                        np.abs(self.loo_ŷ_[sgn, np.newaxis]),
                        np.sign(self.loo_ŷ_[sgn, np.newaxis]),
                    ]
                )
                y_qr = (
                    np.abs(self.loo_residuals_[sgn]) / np.maximum(np.abs(self.loo_ŷ_)[sgn], eps)
                    if "/ŷ" in target_type
                    else np.abs(self.loo_residuals_[sgn])
                )
                quantile_regressors[quantile] = QuantileRegressor(
                    quantile=quantile, alpha=np.sqrt(eps), solver="highs"
                ).fit(X_qr, y_qr)
        # Predict the confidence interval for the nonconformity measure.
        ŷ = self.decision_function(X)
        X_qr = np.hstack(
            [
                self.nonconformity_measure(X)[:, np.newaxis],
                ŷ[:, np.newaxis],
                np.abs(ŷ[:, np.newaxis]),
                np.sign(ŷ[:, np.newaxis]),
            ]
        )
        Δ_lower = np.minimum(
            self.conformal_regressors_["Δ⁻"][quantile].predict(X_qr),
            np.abs(ŷ) * self.conformal_regressors_["Δ⁻/ŷ"][quantile].predict(X_qr),
        )
        Δ_upper = np.minimum(
            self.conformal_regressors_["Δ⁺"][quantile].predict(X_qr),
            np.abs(ŷ) * self.conformal_regressors_["Δ⁺/ŷ"][quantile].predict(X_qr),
        )
        Δ_lower, Δ_upper = np.maximum(0, Δ_lower), np.maximum(0, Δ_upper)
        # Assemble the confidence interval.
        C = np.hstack(((ŷ - Δ_lower)[:, np.newaxis], (ŷ + Δ_upper)[:, np.newaxis]))
        # In case of classification, convert the decision function values to probabilities.
        if self._estimator_type == "classifier":
            C = np.hstack(
                [
                    self.predict_proba_calibrator_.transform(C[:, 0])[:, np.newaxis],
                    self.predict_proba_calibrator_.transform(C[:, 1])[:, np.newaxis],
                ]
            )
            C = np.dstack([1 - C[:, ::-1], C])
        return C

    def decision_function(self, X: FloatMatrix[F]) -> FloatVector[F]:
        """Evaluate this predictor's prediction function."""
        # Compute the point predictions ŷ(X).
        ŷ: FloatVector[F]
        if self.primal_:
            # Apply the feature map φ and predict as ŷ(x) := φ(x)'β̂.
            φ = cast(KernelApproximatingFeatureMap, self.primal_feature_map_).transform(X)
            ŷ = np.real(φ @ self.β̂_)
            ŷ = np.ascontiguousarray(ŷ)
        else:
            # Apply an affine transformation to X, then predict as ŷ(x) := k(x, X) α̂ + 1'α̂.
            X = cast(AffineFeatureMap, self.dual_feature_map_).transform(X)
            K = rbf_kernel(X, self.X_, gamma=0.5)
            b = np.sum(self.α̂_)
            ŷ = K @ self.α̂_ + b
        return ŷ

    def predict(self, X: FloatMatrix[F]) -> GenericVector:
        """Predict the output on a given dataset."""
        # Compute the point predictions ŷ(X).
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
            # The decision function is the point prediction.
            ŷ = ŷ_df
        # Map back to the training target dtype.
        ŷ = ŷ.astype(self.y_dtype_)
        return ŷ

    def predict_proba(
        self,
        X: FloatMatrix[F],
        *,
        confidence_interval: bool = False,
        confidence_level: float = 0.95,
    ) -> FloatVector[F] | FloatMatrix[F] | FloatTensor[F]:
        """Predict the class probability or confidence interval."""
        if confidence_interval:
            # Return the confidence interval for classification or regression.
            C = self.predict_confidence_interval(X, confidence_level=confidence_level)
            return C
        if self._estimator_type == "classifier":
            # Return the class probabilities for classification.
            ŷ_classification = self.decision_function(X)
            p = self.predict_proba_calibrator_.transform(ŷ_classification)
            P = np.hstack([1 - p[:, np.newaxis], p[:, np.newaxis]])
        else:
            # Return the point predictions for regression.
            ŷ_regression = self.predict(X)
            P = ŷ_regression
        return P

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
