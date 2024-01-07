"""Affine separator."""

from typing import TypeVar, cast

import numpy as np
from sklearn.utils import check_consistent_length, check_random_state, check_X_y

from neo_ls_svm._affine_feature_map import AffineFeatureMap
from neo_ls_svm._affine_normalizer import AffineNormalizer
from neo_ls_svm._quantizer import sample_bins_quantized_ecdf
from neo_ls_svm._typing import FloatMatrix, FloatVector

F = TypeVar("F", np.float32, np.float64)


def pairwise_distances(X: FloatMatrix[F], Y: FloatMatrix[F]) -> FloatMatrix[F]:
    """Compute the pairwise squared Euclidian distances between the rows of X and Y."""
    d: FloatMatrix[F] = (
        np.sum(X * X, axis=1, keepdims=True) - 2 * X @ Y.T + np.sum(Y * Y, axis=1, keepdims=True).T
    )
    return d


def nearest_neighbours(X: FloatMatrix[F], Y: FloatMatrix[F]) -> FloatMatrix[F]:
    """Find the rows in Y that are nearest to each row in X."""
    pd = pairwise_distances(X, Y)
    idx = np.argmin(pd, axis=1, keepdims=True)
    nn = np.take_along_axis(Y, idx, axis=0)
    return nn


def _faster_svd(X: FloatMatrix[F]) -> tuple[FloatVector[F], FloatMatrix[F]]:
    """Faster algorithm to compute the the right singular vectors of a matrix.

    This function is equivalent to:

      _, s, VH = np.linalg.svd(X, full_matrices=False)
      V = VH.conj().T
    """
    if X.shape[0] >= X.shape[1]:  # Tall and skinny matrix X.
        e, V = np.linalg.eigh(X.conj().T @ X)
        s = np.sqrt(np.abs(e))[::-1]  # eigh returns eigenvalues in ascending order.
        V = V[:, ::-1]
    else:  # Fat and wide matrix X.
        e, U = np.linalg.eigh(X @ X.conj().T)
        s = np.sqrt(np.abs(e))[::-1]
        U = U[:, ::-1]
        nonzero_sv = s > 0
        s, U = s[nonzero_sv], U[:, nonzero_sv]
        V = (X.conj().T @ U) / s[np.newaxis, :]
    return s, V


class AffineSeparator(AffineNormalizer):
    """Affine separator.

    Applies the transformation (x - shift) @ diag(1 / scale) @ A to the input row x so that:

      1. shift centers the features x so that the class bins are optimally separated,
      2. scale scales the shifted features x so that the difference between two samples from
         different class bins is the separability between those class bins, and
      3. A transforms the shifted and scaled features x to optimally separate the labels y.

    The shift and scale are computed by AffineNormalizer.

    To compute the matrix A, the samples are first quantized into class bins. Then, for each class
    bin we select samples P on the edge of the class bin and their (approximate) nearest neighbours
    Q on the edge of all other class bins (the class bin's complement). We want to find a matrix A
    that separates rows p and q of the matrices P and Q, respectively. In other words, we're looking
    for a matrix A that maximizes the distance ||pA - qA||. If UΣV' is the SVD of P-Q, then the
    first columns of V will maximize that distance. This process is repeated for all of the class
    bins, and the final A is a horizontal concatenation of all right singular vectors.

    Next, we'll optimally scale A for Gaussian kernel-based methods with a scalar λ. If f(λA) is the
    expected squared distance between a pair of inter-bin examples, and g(λA) is the expected
    squared distance of a pair of intra-bin examples, then we know that the Gaussian kernel will
    evaluate to exp(-f(λA)/2) and exp(-g(λA)/2) in these two cases, respectively. We then choose the
    scalar λ so that it optimally separates the inter- and intra-bin samples as follows:

      λ := argmin exp(-f(λA)/2) - exp(-g(λA)/2)
      d/dλ exp(-f(λA)/2) - exp(-g(λA)/2) = 0
      d/dλ exp(-λ²f(A)/2) - exp(-λ²g(A)/2) = 0
      f exp(-λ²f/2) = g exp(-λ²g/2)
      exp(λ²(f-g)/2) = f/g
      λ²(f-g)/2 = log(f/g)
      λ = sqrt(2 log(f/g) / (f-g))
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        append_features: bool = False,
        rank_threshold: float = 2e-2,
        edge_sample_size: int = 384,
        edge_search_multiplier: int = 4,
        random_state: int | np.random.RandomState | None = 42,
    ) -> None:
        self.shift = 0.0
        self.scale = 1.0
        self.A = None
        self.append_features = append_features
        self.rank_threshold = rank_threshold
        self.edge_sample_size = edge_sample_size
        self.edge_search_multiplier = edge_search_multiplier
        self.random_state = random_state

    def fit(
        self,
        X: FloatMatrix[F],
        y: FloatVector[F] | None = None,
        sample_weight: FloatVector[F] | None = None,
    ) -> AffineFeatureMap:
        """Fit this transformer."""
        # Validate X and y.
        assert y is not None
        X, y = check_X_y(X, y, dtype=(np.float64, np.float32))
        y = np.ravel(np.asarray(y)).astype(X.dtype)
        # Learn an optimal shift and scale.
        super().fit(X, y, sample_weight)
        # Transform X with the computed shift and scale.
        X = super().transform(X)  # Note: this won't append features yet because A is still None.
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
        if len(X_bins) <= 1:
            return self
        # Increase sample size if there are only two bins.
        if len(X_bins) == 2:  # noqa: PLR2004
            self.edge_sample_size = int(self.edge_sample_size * 4 / 3)
        # Learn a tranform A that optimally separates each class bin from the other class bins.
        A_bins, X_bins_edge, X_not_bins_edge = [], [], []
        generator = check_random_state(self.random_state)
        for i in range(len(X_bins)):
            # Generate samples from class bin i.
            idx = generator.choice(
                len(X_bins[i]), size=self.edge_sample_size, p=np.ravel(s_bins[i])
            )
            X_bin_i_sample = X_bins[i][idx, :]
            # Generate samples from the complement of class bin i.
            X_not_bin_i = np.vstack([X_bin for j, X_bin in enumerate(X_bins) if j != i])
            s_not_bin_i = np.hstack([sample_weight_[bin] for j, bin in enumerate(bins) if j != i])  # noqa: A001
            idx = generator.choice(
                len(X_not_bin_i),
                size=self.edge_sample_size * self.edge_search_multiplier,
                p=np.ravel(s_not_bin_i) / np.sum(s_not_bin_i),
            )
            X_not_bin_i_sample = X_not_bin_i[idx, :]
            # Find the samples in the complement of bin i that are closest to bin i.
            X_not_bin_i_edge = nearest_neighbours(X_bin_i_sample, X_not_bin_i_sample)
            X_not_bins_edge.append(X_not_bin_i_edge)
            # Find the samples in bin i that are closest to the nearest neighbours in the bin i's
            # complement.
            idx = generator.choice(
                len(X_bins[i]),
                size=self.edge_sample_size * self.edge_search_multiplier,
                p=np.ravel(s_bins[i]),
            )
            X_bin_i_sample = X_bins[i][idx, :]
            X_bin_i_edge = nearest_neighbours(X_not_bin_i_edge, X_bin_i_sample)
            X_bins_edge.append(X_bin_i_edge)
            # Learn the vectors that maximise the separation between samples from the edge of bin i
            # and the edge of its complement.
            s, V = _faster_svd(X_bin_i_edge - X_not_bin_i_edge)
            rank = np.sum(s > self.rank_threshold * s[0])
            A_bins.append(V[:, :rank])  # TODO: Should these be weighted by the singular values?
        self.A_ = np.hstack(A_bins)
        # Scale A optimally to improve performance for Gaussian kernel-based models.
        inter_bin_distance, intra_bin_distance = 0.0, 0.0
        num_inter_pairs = self.edge_sample_size * (self.edge_sample_size + 1) / 2
        num_intra_pairs = self.edge_sample_size * (self.edge_sample_size - 1) / 2
        for X_bin_edge, X_not_bin_edge, n_bin in zip(
            X_bins_edge, X_not_bins_edge, n_bins, strict=True
        ):
            inter_bin_distance += (
                n_bin
                * np.sum(
                    np.tril(pairwise_distances(X_bin_edge @ self.A_, X_not_bin_edge @ self.A_), k=0)
                )
                / num_inter_pairs
            )
            intra_bin_distance += (
                n_bin
                * np.sum(
                    np.tril(pairwise_distances(X_bin_edge @ self.A_, X_bin_edge @ self.A_), k=-1)
                )
                / num_intra_pairs
            )
        inter_bin_distance /= sum(n_bins)
        intra_bin_distance /= sum(n_bins)
        λ = (
            np.sqrt(
                2
                * np.log(inter_bin_distance / intra_bin_distance)
                / (inter_bin_distance - intra_bin_distance)
            )
            if intra_bin_distance > 0
            else 1
        )
        self.A_ *= λ  # TODO: SVM performance is quite sensitive to this scalar. Can it be improved?
        return self
