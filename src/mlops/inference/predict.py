import os
from pathlib import Path

import mlflow
import pandas as pd
import requests


def ingest_data():
    """
    Ingest Test Data

    Returns:
        Dataframe: Pandas df w/ test data
    """
    input_dir = Path(__file__).parent.parent / "data"
    path = input_dir / "test_dataset.parquet"
    df = pd.read_parquet(path).reset_index(drop=True).drop("target", axis=1)
    return df


def load_model():
    """
    Load model from Mlflow

    Returns:
        model obj: Model object from mlflow
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500"))
    experiment_name = os.getenv("EXPERIMENT_NAME", "xgb_best_model")
    print(experiment_name)
    model_uri = f"models:/{experiment_name}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def make_prediction(model, X_test):
    """
    Make a prediction with model

    Args:
        model (model obj): trained model obj
        X_test (Dataframe): test dataframe

    Returns:
        Dataframe: df with prediction column
    """
    prediction = model.predict(X_test)
    X_test["prediction"] = (prediction > 0.5).astype(int)
    return X_test


def score_predictions(row:pd.Series):
    """
    Score predictions

    Args:
        row (pd.Series): Pandas series

    Returns:
        json: json with prediction outcome
    """
    url = "http://localhost:8080/predict"
    row_as_dict = row.to_dict()
    response = requests.post(
        url, json=row_as_dict,
        timeout=10
    )  ## post the information in json format
    try:
        result = response.json()  ## get the server response
        print(result)
        return result
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    return None


if __name__ == "__main__":
    df_test = ingest_data()
    # model = load_model()
    for i in range(len(df_test)):
        print(f"Observation: {i}")
        score_predictions(df_test.iloc[[i]])
