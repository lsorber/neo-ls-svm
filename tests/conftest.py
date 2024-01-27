"""Test fixtures."""

from typing import TypeAlias

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
from _pytest.fixtures import SubRequest
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_union
from sklearn.preprocessing import OneHotEncoder
from skrub import TableVectorizer

Dataset: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


@pytest.fixture(
    params=[
        pytest.param(
            43926,
            id="dataset:ames_housing",  # Regression
        ),
        pytest.param(
            531,
            id="dataset:boston",  # Regression
        ),
        pytest.param(
            287,
            id="dataset:wine_quality",  # Regression
        ),
        pytest.param(
            40945,
            id="dataset:titanic",  # Binary classification
        ),
        pytest.param(
            31,
            id="dataset:credit-g",  # Binary classification
        ),
        pytest.param(
            54,
            id="dataset:vehicle",  # Multiclass classification
        ),
    ],
)
def dataset(request: SubRequest) -> Dataset:
    """Train and test dataset fixture."""
    # Download the dataset.
    X, y = sklearn.datasets.fetch_openml(
        data_id=request.param, return_X_y=True, as_frame=True, parser="auto"
    )
    # Split in train and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def table_vectorizer() -> TableVectorizer:
    """Robust TableVectorizer."""
    # Fix missing values passing through.
    numerical_transformer = make_union(
        SimpleImputer(missing_values=np.nan, strategy="median"),
        MissingIndicator(
            missing_values=np.nan,
            features="missing-only",
            sparse=False,
            error_on_new=False,
        ),
    )
    # Fix OHE generating warnings for unknown categories.
    ohe = OneHotEncoder(drop=None, handle_unknown="infrequent_if_exist")
    # Create the TableVectorizer.
    return TableVectorizer(
        numerical_transformer=numerical_transformer, low_cardinality_transformer=ohe
    )
