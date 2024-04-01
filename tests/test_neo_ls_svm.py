"""Test Neo LS-SVM."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.utils.estimator_checks import check_estimator
from skrub import TableVectorizer

from neo_ls_svm import NeoLSSVM
from tests.conftest import Dataset


def test_compare_neo_ls_svm_with_svm(dataset: Dataset, table_vectorizer: TableVectorizer) -> None:
    """Compare Neo LS-SVM with SVM."""
    # Split the dataset.
    X_train, X_test, y_train, y_test = train_test_split(*dataset, test_size=0.15, random_state=42)
    # Create the pipelines.
    num_unique = len(y_train.unique())
    binary = num_unique == 2  # noqa: PLR2004
    multiclass = 2 < num_unique <= np.ceil(np.sqrt(len(y_train)))  # noqa: PLR2004
    if binary:
        neo_ls_svm_pipeline = make_pipeline(table_vectorizer, NeoLSSVM())
        svm_pipeline = make_pipeline(table_vectorizer, SVC())
    elif multiclass:
        neo_ls_svm_pipeline = make_pipeline(table_vectorizer, OneVsRestClassifier(NeoLSSVM()))
        svm_pipeline = make_pipeline(table_vectorizer, OneVsRestClassifier(SVC()))
    else:
        neo_ls_svm_pipeline = make_pipeline(table_vectorizer, NeoLSSVM())
        svm_pipeline = make_pipeline(table_vectorizer, SVR())
    # Train the models.
    neo_ls_svm_pipeline.fit(X_train, y_train)
    svm_pipeline.fit(X_train, y_train)
    # Compare the results.
    neo_ls_svm_score = neo_ls_svm_pipeline.score(X_test, y_test)
    svm_score = svm_pipeline.score(X_test, y_test)
    assert neo_ls_svm_score > svm_score
    if multiclass:
        return
    # Verify the coherence of the predicted quantiles.
    ŷ_test_quantiles = neo_ls_svm_pipeline.predict(X_test, quantiles=np.linspace(0.1, 0.9, 3))
    for j in range(ŷ_test_quantiles.shape[1] - 1):
        if binary:
            for k in range(ŷ_test_quantiles.shape[2]):
                assert np.all(ŷ_test_quantiles[:, j, k] <= ŷ_test_quantiles[:, j + 1, k])
        else:
            assert np.all(ŷ_test_quantiles[:, j] <= ŷ_test_quantiles[:, j + 1])
    # Verify the coverage of the predicted intervals.
    for desired_coverage in (0.7, 0.8, 0.9, 0.95):
        ŷ_test_interval = neo_ls_svm_pipeline.predict(X_test, coverage=desired_coverage)
        if binary:
            assert np.all(ŷ_test_interval >= 0)
            assert np.all(ŷ_test_interval <= 1)
            assert np.all(ŷ_test_interval[:, 0, 0] <= ŷ_test_interval[:, 1, 0])
            assert np.all(ŷ_test_interval[:, 0, 1] <= ŷ_test_interval[:, 1, 1])
            is_neg = y_test == neo_ls_svm_pipeline.steps[-1][1].classes_[0]
            is_pos = ~is_neg
            neg_covered = np.any(ŷ_test_interval[:, :, 0] > 0.5, axis=1) & is_neg  # noqa: PLR2004
            pos_covered = np.any(ŷ_test_interval[:, :, 1] > 0.5, axis=1) & is_pos  # noqa: PLR2004
            covered = neg_covered | pos_covered
        else:
            assert np.all(ŷ_test_interval[:, 0] <= ŷ_test_interval[:, 1])
            covered = (ŷ_test_interval[:, 0] <= y_test) & (y_test <= ŷ_test_interval[:, 1])
        actual_coverage = np.mean(covered)
        assert actual_coverage >= 0.97 * desired_coverage


def test_pandas_support(dataset: Dataset) -> None:
    """Test pandas support."""
    # Unpack the dataset.
    X, y = dataset
    X = pd.get_dummies(X).fillna(0)  # Neo LS-SVM only supports finite numerical input.
    # Split the dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    # Skip on multiclass problems.
    num_unique = len(y_train.unique())
    binary = num_unique == 2  # noqa: PLR2004
    multiclass = 2 < num_unique <= np.ceil(np.sqrt(len(y_train)))  # noqa: PLR2004
    if multiclass:
        return
    # Train a Neo LS-SVM model.
    model = NeoLSSVM()
    model.fit(X_train, y_train)
    # Test pandas support of methods that output vectors.
    vector_methods = ["decision_function", "predict", "predict_std"]
    for method_name in vector_methods + ([] if binary else ["predict_proba"]):
        method = getattr(model, method_name)
        ŷ_test_np = method(np.asarray(X_test))
        ŷ_test_pd = method(X_test)
        assert isinstance(ŷ_test_pd, pd.Series)
        assert np.all(np.asarray(ŷ_test_pd) == ŷ_test_np)
        assert ŷ_test_pd.index.equals(X_test.index)
    # Test pandas support of methods that output matrices or tensors.
    tensor_methods = ["predict_quantiles", "predict_interval"]
    for method_name in tensor_methods + (["predict_proba"] if binary else []):
        method = getattr(model, method_name)
        ŷ_quantiles_test_np = method(np.asarray(X_test))
        ŷ_quantiles_test_pd = method(X_test)
        is_tensor = ŷ_quantiles_test_np.ndim == 3  # noqa: PLR2004
        if is_tensor:
            ŷ_quantiles_test_np = np.vstack(
                [ŷ_quantiles_test_np[:, :, k] for k in range(ŷ_quantiles_test_np.shape[2])]
            )
        assert isinstance(ŷ_quantiles_test_pd, pd.DataFrame)
        assert np.all(np.asarray(ŷ_quantiles_test_pd) == ŷ_quantiles_test_np)
        assert is_tensor or ŷ_quantiles_test_pd.index.equals(X_test.index)


def test_sklearn_check_estimator() -> None:
    """Check that the model conforms to sklearn's standards."""
    model = NeoLSSVM(estimator_type="regressor")
    check_estimator(model)
    model = NeoLSSVM(estimator_type="classifier")
    check_estimator(model)
