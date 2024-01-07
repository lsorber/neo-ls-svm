"""Test Neo LS-SVM on datasets."""

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.utils.estimator_checks import check_estimator
from skrub import TableVectorizer

from neo_ls_svm import NeoLSSVM
from tests.conftest import Dataset


def test_compare_neo_ls_svm_with_svm(dataset: Dataset, table_vectorizer: TableVectorizer) -> None:
    """Compare Neo LS-SVM with SVM."""
    # Unpack the dataset.
    X_train, X_test, y_train, y_test = dataset
    # Create the pipelines.
    num_unique = len(y_train.unique())
    if num_unique == 2:  # noqa: PLR2004
        neo_ls_svm_pipeline = make_pipeline(table_vectorizer, NeoLSSVM())
        svm_pipeline = make_pipeline(table_vectorizer, SVC())
    elif num_unique <= np.ceil(np.sqrt(len(y_train))):
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


def test_sklearn_check_estimator() -> None:
    """Check that the model conforms to sklearn's standards."""
    model = NeoLSSVM(estimator_type="regressor")
    check_estimator(model)
    model = NeoLSSVM(estimator_type="classifier")
    check_estimator(model)
