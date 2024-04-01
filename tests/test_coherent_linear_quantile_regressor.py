"""Test the Coherent Linear Quantile Regressor."""

from sklearn.utils.estimator_checks import check_estimator

from neo_ls_svm._coherent_linear_quantile_regressor import CoherentLinearQuantileRegressor


def test_sklearn_check_estimator() -> None:
    """Check that the meta-estimator conforms to sklearn's standards."""
    model = CoherentLinearQuantileRegressor(quantiles=(0.5,))
    check_estimator(model)
