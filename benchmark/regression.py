"""Regression benchmark."""  # noqa: INP001

import time

import numpy as np
import pandas as pd
import sklearn.datasets
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from skrub import TableVectorizer

from neo_ls_svm import NeoLSSVM

# AutoML regression suite [1].
# [1] https://arxiv.org/pdf/2207.12560.pdf
regression_tasks = [
    ("abalone", 5),
    ("Airlines_DepDelay_10M", 1),  # Large
    ("Allstate_Claims_Severity", 1),  # Large
    ("black_friday", 1),
    ("boston", 1),
    ("Brazilian_houses", 4),
    ("Buzzinsocialmedia_Twitter", 1),  # Large
    ("colleges", 14),
    ("diamonds", 1),
    ("elevators", 1),
    ("house_16H", 1),
    ("house_prices_nominal", 1),
    ("house_sales", 3),
    ("Mercedes_Benz_Greener_Manufacturing", 2),
    ("MIP-2016-regression", 3),
    ("Moneyball", 2),
    ("nyc-taxi-green-dec-2016", 3),  # Large
    ("OnlineNewsPopularity", 2),  # Large
    ("pol", 1),
    ("QSAR-TID-10980", 1),  # Large
    ("QSAR-TID-11", 1),  # Large
    ("quake", 2),
    ("Santander_transaction_value", 1),  # Large
    ("SAT11-HAND-runtime-regression", 1),
    ("sensory", 1),
    ("socmob", 1),
    ("space_ga", 1),
    ("tecator", 1),
    ("topo_2_1", 1),
    ("us_crime", 2),
    ("wine_quality", 1),
    ("Yolanda", 2),  # Large
    ("yprop_4_1", 1),
]

# Create the regression models.
num = make_union(SimpleImputer(strategy="median"), MissingIndicator(error_on_new=False))
ohe = OneHotEncoder(drop=None, handle_unknown="infrequent_if_exist")
table_vectorizer = TableVectorizer(numerical_transformer=num, low_cardinality_transformer=ohe)
models = [
    {"name": "SVR", "pipeline": make_pipeline(clone(table_vectorizer), SVR())},
    {
        "name": "NeoLSSVM",
        "pipeline": make_pipeline(clone(table_vectorizer), NeoLSSVM(estimator_type="regressor")),
    },
    {"name": "LGBMRegressor", "pipeline": LGBMRegressor()},
]

# Benchmark the models on the regression tasks.
regression_records = []
for i, (task_id, task_version) in enumerate(regression_tasks):
    # Download the dataset.
    try:
        X, y = sklearn.datasets.fetch_openml(
            task_id, version=task_version, return_X_y=True, as_frame=True, parser="auto"
        )
    except Exception as e:  # noqa: BLE001
        print(f"Skipping {task_id} because of {e}...")  # noqa: T201
        continue
    # Split in train and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    # Benchmark each model.
    for model in models:
        # Exit early if the dataset is too large.
        if X_train.size > 1_000_000:  # noqa: PLR2004
            print(f"Skipping {task_id} because it's too large...")  # noqa: T201
            continue
        # Exit early if SVR is too slow.
        if model["name"] == "SVR" and len(X_train) > 9999:  # noqa: PLR2004
            print(f"Skipping {task_id} because it's too large for SVM...")  # noqa: T201
            continue
        # Clone the model.
        pipeline = clone(model["pipeline"])
        # Warm up the model.
        if i == 0:
            pipeline.fit(X_train, y_train)
            pipeline = clone(model["pipeline"])
        # Fit the model.
        t1 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        t2 = time.perf_counter()
        # Measure the model's performance.
        score = pipeline.score(X_test, y_test)
        # Store the results.
        record = {"dataset": task_id, "model": model["name"], "score": score, "time": t2 - t1}
        regression_records.append(record)
        print(record)  # noqa: T201
regression_df = pd.DataFrame(regression_records)

# Create a comparison table.
score_df = regression_df.pivot(index="dataset", columns="model", values="score")
time_df = regression_df.pivot(index="dataset", columns="model", values="time")
comparison_records = []
for task_id, row in score_df.iterrows():
    sorted_models = row.sort_values(ascending=False)
    best_model, second_best_model = sorted_models.index[:2]
    record = {"dataset": task_id.lower()}
    for model_name in score_df.columns:
        score = 100 * row[model_name]
        ptime = time_df.at[task_id, model_name]
        value = f"{score:.1f}% ({ptime:.1f}s)" if not np.isnan(score) else "/"
        if model_name == best_model:
            value = f"🥇 {value}"
        elif model_name == second_best_model:
            value = f"🥈 {value}"
        record[model_name] = value
    comparison_records.append(record)
comparison_df = (
    pd.DataFrame(comparison_records).set_index("dataset").sort_index(axis=0).sort_index(axis=1)
)
print(comparison_df.to_markdown(stralign="right"))  # noqa: T201
