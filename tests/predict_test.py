import os

import numpy as np
import pandas as pd

from src.mlops.inference.predict import (
    load_model,
    make_prediction,
    score_predictions
)


def test_load_model(mocker):
    # Create patch for mlflow load fn & model mock
    mock_mlflow = mocker.patch("mlflow.pyfunc.load_model")
    mock_model = mocker.Mock()
    mock_mlflow.return_value = mock_model
    # Mock env vars
    mocker.patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://fake-tracking-uri",
            "EXPERIMENT_NAME": "fake_experiment",
        },
    )

    model = load_model()
    mock_mlflow.assert_called_once_with("models:/fake_experiment/latest")
    assert model == mock_model


def test_make_prediction(mocker):
    X_test = pd.DataFrame(
        {
            "age": [52],
            "sex": [1],
            "cp": [0],
            "trestbps": [125],
            "chol": [212],
            "fbs": [0],
            "restecg": [1],
            "thalach": [168],
            "exang": [0],
            "oldpeak": [1.0],
            "slope": [2],
            "ca": [2],
            "thal": [3],
        }
    )

    mock_model = mocker.Mock()
    mock_model.predict.return_value = np.array([0.3])

    result = make_prediction(mock_model, X_test.copy())

    expected = pd.Series([0], name="prediction")
    pd.testing.assert_series_equal(result["prediction"], expected)


def test_score_predictions_success(mocker):

    # Create a dummy input row
    row = pd.Series(
        {
            "age": [52],
            "sex": [1],
            "cp": [0],
            "trestbps": [125],
            "chol": [212],
            "fbs": [0],
            "restecg": [1],
            "thalach": [168],
            "exang": [0],
            "oldpeak": [1.0],
            "slope": [2],
            "ca": [2],
            "thal": [3],
        }
    )

    # Mock the response from requests.post
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": 0}

    # Patch requests.post to return the mock response
    mock_post = mocker.patch("requests.post", return_value=mock_response)

    # Call the function
    result = score_predictions(row)

    # Assert requests.post was called correctly
    mock_post.assert_called_once_with(
        "http://localhost:8080/predict", json=row.to_dict()
    )

    # Assert .json() was called on the response
    mock_response.json.assert_called_once()

    # Assert the returned result is as expected
    assert result == {"prediction": 0}
