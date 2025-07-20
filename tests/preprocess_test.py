import pytest
import pandas as pd
from src.mlops.processing.prepare_features import map_data_types
from src.mlops.processing.preprocess import Preprocessor


@pytest.fixture(name="dataframe")
def setup_data_frame():
    input_data = {
        "age": [52, 53, 70, 61, 62, 58, 58, 55, 46, 54],
        "sex": [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        "cp": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "trestbps": [125, 140, 145, 148, 138, 100, 114, 160, 120, 122],
        "chol": [212, 203, 174, 203, 294, 248, 318, 289, 249, 286],
        "fbs": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "restecg": [1, 0, 1, 1, 1, 0, 2, 0, 0, 0],
        "thalach": [168, 155, 125, 161, 106, 122, 140, 145, 144, 116],
        "exang": [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
        "oldpeak": [1.0, 3.1, 2.6, 0.0, 1.9, 1.0, 4.4, 0.8, 0.8, 3.2],
        "slope": [2, 0, 0, 2, 1, 1, 0, 1, 2, 1],
        "ca": [2, 0, 0, 1, 3, 0, 3, 1, 0, 2],
        "thal": [3, 3, 3, 3, 2, 2, 1, 3, 3, 2],
        "target": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    }
    df_input = pd.DataFrame(input_data)
    return df_input


def test_identify_categorical_vars(dataframe):
    # Instance Preprocessor class
    preprocess = Preprocessor(data=dataframe, target="target")
    preprocess.identify_categorical_encoded_vars()
    # Check
    expected = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    actual = preprocess.categorical_vars
    assert expected == actual


def test_identify_numerical_vars(dataframe):
    num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    # Instance Preprocessor class
    preprocess = Preprocessor(data=dataframe, target="target")
    preprocess.categorical_vars = cat_features
    preprocess.identify_numerical_vars()
    # Check
    expected = num_features + ["target"]
    actual = preprocess.numerical_vars
    print("actual", actual)
    assert expected == actual


def test_split_dataset(dataframe):
    # The output should be 60% train, 20% val and 20% test
    train_size = 6
    val_size = 2
    test_size = 2
    # Instance Preprocessor class
    preprocess = Preprocessor(data=dataframe, target="target")
    preprocess.build_datasets()
    # Checks
    expected = (train_size, val_size, test_size)
    actual = (
        preprocess.data_dict["x_train"].shape[0],
        preprocess.data_dict["x_valid"].shape[0],
        preprocess.df_test.shape[0],
    )

    assert expected == actual


def test_correct_data_types(dataframe):
    df = dataframe.drop("target", axis=1)
    df = map_data_types(df)

    expected = {
        "age": "int64",
        "sex": "int64",
        "cp": "int64",
        "trestbps": "int64",
        "chol": "int64",
        "fbs": "int64",
        "restecg": "int64",
        "thalach": "int64",
        "exang": "int64",
        "oldpeak": "float64",
        "slope": "int64",
        "ca": "int64",
        "thal": "int64",
    }

    actual = {col: df[col].dtype.name for col in df.columns}
    print(expected)
    print(actual)
    assert expected == actual
