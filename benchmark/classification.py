"""Classification benchmark."""  # noqa: INP001

import time

import numpy as np
import pandas as pd
import sklearn.datasets
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from skrub import TableVectorizer

from neo_ls_svm import NeoLSSVM

# AutoML binary classification suite [1].
# [1] https://arxiv.org/pdf/2207.12560.pdf
classification_tasks = [
    ("ada", 1),
    ("adult", 4),
    ("airlines", 1),
    # ("albert", 1),  # Doesn't download?  # noqa: ERA001
    ("Amazon_employee_access", 1),
    ("APSFailure", 1),
    ("arcene", 2),
    ("Australian", 4),
    ("bank-marketing", 1),
    ("Bioresponse", 1),
    ("blood-transfusion-service-center", 1),
    ("christine", 1),
    ("churn", 1),
    ("Click_prediction_small", 10),
    ("gina", 1),
    ("guillermo", 1),
    ("Higgs", 3),
    ("Internet-Advertisements", 2),
    ("jasmine", 1),
    ("kc1", 1),
    # ("KDDCup09-Upselling", 3),  # Crashes?  # noqa: ERA001
    ("KDDCup09_appetency", 1),
    ("kick", 1),
    ("kr-vs-kp", 1),
    ("madeline", 1),
    ("MiniBooNE", 1),
    ("nomao", 1),
    ("numerai28.6", 2),
    ("ozone-level-8hr", 1),
    ("pc4", 1),
    ("philippine", 1),
    ("PhishingWebsites", 1),
    ("phoneme", 1),
    ("porto-seguro", 3),
    ("qsar-biodeg", 1),
    ("riccardo", 1),
    ("Satellite", 1),
    ("sf-police-incidents", 6),
    ("sylvine", 1),
    ("wilt", 2),
]

# Create the classification models.
num = make_union(SimpleImputer(strategy="median"), MissingIndicator(error_on_new=False))
ohe = OneHotEncoder(drop=None, handle_unknown="infrequent_if_exist")
table_vectorizer = TableVectorizer(numerical_transformer=num, low_cardinality_transformer=ohe)
models = [
    {"name": "SVC", "pipeline": make_pipeline(clone(table_vectorizer), SVC(probability=True))},
    {
        "name": "NeoLSSVM",
        "pipeline": make_pipeline(clone(table_vectorizer), NeoLSSVM(estimator_type="classifier")),
    },
    {"name": "LGBMClassifier", "pipeline": LGBMClassifier()},
]

# Benchmark the models on the classification tasks.
classification_records = []
for i, (task_id, task_version) in enumerate(classification_tasks):
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
        print(f"Training on {task_id} with {model['name']} (shape={X_train.shape})...")  # noqa: T201
        # Exit early if the dataset is too large.
        if X_train.size > 1_000_000:  # noqa: PLR2004
            print(f"Skipping {task_id} because it's too large...")  # noqa: T201
            continue
        # Exit early if SVR is too slow.
        if model["name"] == "SVC" and len(X_train) > 9999:  # noqa: PLR2004
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
        y_test_score = pipeline.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_test_score)
        # Store the results.
        record = {"dataset": task_id, "model": model["name"], "score": score, "time": t2 - t1}
        classification_records.append(record)
        print(record)  # noqa: T201
classification_df = pd.DataFrame(classification_records)

# Create a comparison table.
score_df = classification_df.pivot(index="dataset", columns="model", values="score")
time_df = classification_df.pivot(index="dataset", columns="model", values="time")
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
