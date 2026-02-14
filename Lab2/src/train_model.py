import argparse
import datetime
import os
import pickle
import random
import sys

import mlflow
import mlflow.sklearn
from joblib import dump

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(".."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp", type=str, required=True, help="Timestamp from GitHub Actions"
    )
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    n_samples = random.randint(200, 2000)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=True,
    )

    os.makedirs("data", exist_ok=True)
    with open("data/data.pickle", "wb") as f:
        pickle.dump(X, f)
    with open("data/target.pickle", "wb") as f:
        pickle.dump(y, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "SyntheticBinaryClassification"
    experiment_name = f"{dataset_name}"  # stable name, don’t spam new experiments
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{dataset_name}_{timestamp}"):
        params = {
            "dataset_name": dataset_name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "model_type": "LogisticRegression",
            "timestamp": timestamp,
        }
        mlflow.log_params(params)

        model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mlflow.log_metrics(
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }
        )

        os.makedirs("models", exist_ok=True)
        model_version = f"model_{timestamp}"
        model_path = os.path.join("models", f"{model_version}_logreg.joblib")
        dump(model, model_path)

        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")
