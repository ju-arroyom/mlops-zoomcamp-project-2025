import os
import mlflow
import requests
import pandas as pd
from pathlib import Path
from mlops.monitoring.metrics_calculation import calculate_metrics


def ingest_data():
    input_dir = Path(__file__).parent.parent / "data"
    path = input_dir / "test_dataset.parquet"
    df = pd.read_parquet(path).reset_index(drop=True).drop("target", axis=1)
    return df


def load_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500"))
    experiment_name = os.getenv("EXPERIMENT_NAME", "xgb_best_model")
    print(experiment_name)
    model_uri = f"models:/{experiment_name}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def make_prediction(model, X_test):
    prediction = model.predict(X_test)
    X_test["prediction"] = (prediction > 0.5).astype(int)
    return X_test


def score_predictions(row):
    url = "http://localhost:8080/predict"
    row_as_dict = row.to_dict()
    response = requests.post(
        url, json=row_as_dict
    )  ## post the information in json format
    if response.status_code == 200:
        result = response.json()  ## get the server response
        print(result)
        return result


if __name__ == "__main__":
    df_test = ingest_data()
    # model = load_model()
    for i in range(len(df_test)):
        print(f"Observation: {i}")
        score_predictions(df_test.iloc[[i]])
