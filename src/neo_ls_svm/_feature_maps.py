"""Random feature maps.

This module implements feature maps φ: Rᵈ -> Cᴰ so that φ(x)'φ(y) approximates the Gaussian kernel
k(x, y) := exp(- ||x - y||² / (2σ²)), or more generally k(x, y) := exp(- ||A(x - y)||² / 2) for a
matrix A.

These approximating feature maps also include transformers to automatically learn an affine
transformation A that optimally separates the examples.

Notable kernel-approximating feature maps are Random Fourier Features [1][2][3], Orthogonal Random
Features [4], and Spherical Structured Features [5]. For a full overview, see the survey [6].

[1] https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
[2] https://arxiv.org/abs/1408.3060
[3] https://gregorygundersen.com/blog/2019/12/23/random-fourier-features
[4] https://arxiv.org/abs/1610.09072
[5] http://proceedings.mlr.press/v70/lyu17a/lyu17a.pdf
[6] https://arxiv.org/abs/2004.11154
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar

import numba
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

from neo_ls_svm._affine_feature_map import AffineFeatureMap
from neo_ls_svm._affine_separator import AffineSeparator
from neo_ls_svm._typing import ComplexMatrix, FloatMatrix, FloatVector

C = TypeVar("C", np.complex64, np.complex128)
F = TypeVar("F", np.float32, np.float64)


@numba.jit(nopython=True, nogil=True, parallel=True, fastmath=True, cache=True)
def _ztz_prod_sinc_zmz(Z: FloatMatrix[F], *, fast_approx: bool = False) -> FloatMatrix[F]:
    """Compute 1/d Z'Z * [Πₖ sinc(Zₖᵢ - Zₖⱼ)]ᵢⱼ)."""
    d, D = Z.shape
    if fast_approx:
        # Slower approx: np.diag(np.sum(Z * Z, axis=0)) / d = np.diag(np.diag(Z.T @ Z)) / d.
        return np.eye(D, dtype=Z.dtype)
    C = Z.T @ Z
    eps = np.finfo(Z.dtype).eps
    for k in range(d):
        for i in numba.prange(D):
            for j in numba.prange(i):
                dz = Z[k, i] - Z[k, j]
                if np.abs(dz) > eps:
                    C[i, j] *= np.sin(dz) / dz
    C_scaled: FloatMatrix[F] = (np.tril(C) + np.tril(C, -1).T) / d
    return C_scaled


class KernelApproximatingFeatureMap(ABC, BaseEstimator, TransformerMixin):
    """Abstract kernel-approximating feature map."""

    def __init__(
        self,
        affine_feature_map: AffineFeatureMap | None = None,
        num_features: int = 512,
        random_state: int | np.random.RandomState | None = 42,
    ):
        self.num_features, self.D = num_features, num_features
        self.affine_feature_map = affine_feature_map or AffineSeparator()
        self.random_state = random_state

    @cached_property
    @abstractmethod
    def complexity_matrix(self) -> FloatMatrix[F]:
        """Compute the complexity regularisation matrix.

        This regularisation matrix is a complement to the classic maximal margin (classification) or
        ridge regression (regression) identity matrix. The complexity matrix directly penalises the
        complexity of the prediction surface φ(x)'w + b by integrating the norm of its normal vector
        ∇ₓφ(x)'w over the normalised feature space (-1, 1)ᵈ:

            φ(x) := exp(1j Z'x) / √D
            ∇ₓφ(x) = diag(exp(1j Z'x)) Z' / √D
            ∇ₓφ(x)∇ₓφ(x)' = diag(exp(1j Z'x)) Z'Z diag(exp(-1j Z'x)) / D

            ∫||∇ₓφ(x)'w||²dx
            = ∫ w'∇ₓφ(x)∇ₓφ(x)'w dx
            = w' ∫∇ₓφ(x)∇ₓφ(x)'dx w
            = w' ∫ Z'Z/D * [exp(1j Z₌ᵢ'x - 1j Z₌ⱼ'x)]ᵢⱼ dx  where  * is the Hadamard product
            = w' (Z'Z/D * ∫ [exp(1j (Z₌ᵢ - Z₌ⱼ)'x)]ᵢⱼ dx) w  where  Z₌ᵢ is the i'th column of Z
            = w' (Z'Z * [2ᵈ/D Πₖ sinc(Zₖᵢ - Zₖⱼ)]ᵢⱼ) w
            ~ w' (Z'Z * [Πₖ sinc(Zₖᵢ - Zₖⱼ)]ᵢⱼ) w

        We also extend the complexity matrix with an entry on the diagonal to also shrink the bias
        term b, and normalise the matrix so that it is invariant to the number of input features d
        and the number of output features D.
        """
        ...

    @abstractmethod
    def fit(
        self,
        X: FloatMatrix[F],
        y: FloatVector[F] | None = None,
        sample_weight: FloatVector[F] | None = None,
    ) -> "KernelApproximatingFeatureMap":
        """Fit this transformer."""
        self.affine_feature_map.fit(X, y, sample_weight)
        self.n_features_in_ = X.shape[1]
        return self

    @abstractmethod
    def transform(self, X: FloatMatrix[F]) -> ComplexMatrix[C]:
        """Transform the given data with this transformer."""
        ...


class RandomFourierFeatures(KernelApproximatingFeatureMap):
    """Random Fourier Features."""

    @classmethod
    def _fourier_features(
        cls, d: int, D: int, dtype: npt.DTypeLike, random_state: int | np.random.RandomState | None
    ) -> FloatMatrix[F]:
        # Draw elements of Z ∈ Rᵈˣᴰ from N(0, 1).
        generator = check_random_state(random_state)
        Z: FloatMatrix[F] = generator.randn(d, D).astype(dtype)
        return Z

    @cached_property
    def complexity_matrix(self) -> FloatMatrix[F]:
        """Compute this feature map's complexity matrix."""
        # Compute the complexity matrix using a fast diagonal approximation.
        C: FloatMatrix[F] = np.eye(self.D + 1, dtype=self.Z_.dtype)
        C[:-1, :-1] = _ztz_prod_sinc_zmz(self.Z_, fast_approx=True)
        return C

    def fit(
        self,
        X: FloatMatrix[F],
        y: FloatVector[F] | None = None,
        sample_weight: FloatVector[F] | None = None,
    ) -> "RandomFourierFeatures":
        """Fit this transformer."""
        # Fit an affine transform.
        super().fit(X, y, sample_weight)
        # Create a linear transformation Z ∈ Rᵈˣᴰ and update the affine transform.
        A = getattr(self.affine_feature_map, "A_", self.affine_feature_map.A)
        d = A.shape[1] if A is not None else X.shape[1]
        self.Z_: FloatMatrix[F] = self._fourier_features(d, self.D, X.dtype, self.random_state)
        self.affine_feature_map.A_ = A @ self.Z_ if A is not None else self.Z_
        return self

    def transform(self, X: FloatMatrix[F]) -> ComplexMatrix[C]:
        """Transform a feature matrix X ∈ Rⁿˣᵈ into φ(X) ∈ Cⁿˣᴰ⁺¹ so that φ(X)ᵢ := [φ(xᵢ)' 1].

        Notice that we can choose to solve an LS-SVM in the primal or dual space using the matrix
        identity (γI + AB)⁻¹ A = A (γI + BA)⁻¹:

            argmin ||y - φ(X)β̂||² + γ||β̂||²
            = (γI + φ(X)'φ(X))⁻¹ φ(X)'y
            = φ(X)' (γI + φ(X)φ(X)')⁻¹y  with identity (γI + AB)⁻¹ A = A (γI + BA)⁻¹
            = φ(X)'a  where  a = (γI + φ(X)φ(X)')⁻¹y = (γI + k(xᵢ, xⱼ))⁻¹y

        This means that k(x, y) = φ(x)'φ(y) by definition. Now we look for a φ(x) so that k(x, y) =
        φ(x)'φ(y) for the Gaussian kernel k(x, y) = exp(- ||y - x||² / 2). If we take h(x) :=
        exp(1j ω'Re{x}) for a standard normally distributed variable ω, then we see that [1]:

            = Eω[h(x)'h(y)]
            = Eω[exp(-1j Re{x}'ω) exp(1j ω'Re{y})]
            = ∫ p(ω) exp(-1j/2 (ω'x + x'ω)) exp(1j/2 (ω'y + y'ω)) dω
            = ∫ p(ω) exp(1j/2 (ω'y + y'ω - x'ω - ω'x)) dω
            = ∫ p(ω) exp(-1/2 [- 1j ω'(y - x) - 1j (y - x)'ω]) dω
            = (2π)^(-D/2) ∫ exp(- ||ω||²/2) exp(-1/2 [- 1j ω'(y - x) - 1j(y - x)'ω]) dω
            = (2π)^(-D/2) ∫ exp(-1/2 [ω'ω - 1j ω'(y - x) - 1j(y - x)'ω]) dω
            = (2π)^(-D/2)
              ∫ exp(-1/2 [ω'ω - 1j ω'(y - x) - 1j(y - x)'ω - ||y - x||²] - 1/2 ||y - x||²) dω
            = (2π)^(-D/2) exp(- ||y - x||² / 2) ∫ exp(-1/2 (ω - 1j(y - x))'(ω - 1j(y - x))) dω
            = exp(- ||y - x||² / 2)

        Notes
        -----
        1. The resulting features are complex-valued. There does exist a real version
           [cos(XZ) sin(XZ)] / √D of this feature map. However, this doubles the size of the linear
           system, which results in an increase of 2³ = 8 work compared to the factor of ~4 for
           working with a complex-valued system of half the size.
        2. The rows of the output φ(X) are of the form [φ(xᵢ)' 1] where x is a row of X transposed
           into a column vector and φ(x) := exp(1j Z'x) / √D so that the model f(x) := φ(x)'w + b
           can be computed for multiple examples X as φ(X)[w; b].

        References
        ----------
        [1] https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/#a1-gaussian-kernel-derivation
        """
        # Apply an affine transformation to X.
        X = self.affine_feature_map.transform(X)
        # Apply the complex feature map to X.
        φ: ComplexMatrix[C] = np.empty(
            (X.shape[0], self.D + 1),
            dtype=np.complex64 if X.dtype == np.float32 else np.complex128,
        )
        φ[:, :-1] = np.exp(-1j * X, dtype=φ.dtype) / np.sqrt(self.D)
        φ[:, -1] = 1
        return φ


class OrthogonalRandomFourierFeatures(RandomFourierFeatures):
    """Orthogonal Random Fourier Features."""

    @classmethod
    def _fourier_features(
        cls, d: int, D: int, dtype: npt.DTypeLike, random_state: int | np.random.RandomState | None
    ) -> FloatMatrix[F]:
        # Draw elements of Z ∈ Rᵈˣᴰ from N(0, 1).
        generator = check_random_state(random_state)
        Z: FloatMatrix[F] = generator.randn(d, D).astype(dtype)
        # Then orthonormalise its columns.
        for j in range(0, D, d):
            Q, _ = np.linalg.qr(Z[:, j : j + d])
            Z[:, j : j + d] = Q
        # And compensate for the loss of magnitude relative to standard normal vectors.
        S = np.sqrt(generator.chisquare(d, size=(1, Z.shape[1])).astype(dtype))
        Z *= S
        return Z
