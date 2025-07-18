import pandas as pd
from src.mlops.processing.prepare_features import map_data_types


def test_correct_data_types():
    input = {'age': 52.0,
            'sex': 1.0,
            'cp': 0.0,
            'trestbps': 125.0,
            'chol': 212.0,
            'fbs': 0.0,
            'restecg': 1.0,
            'thalach': 168.0,
            'exang': 0.0,
            'oldpeak': 1.0,
            'slope': 2.0,
            'ca': 2.0,
            'thal': 3.0}
    df = pd.DataFrame([input])
    df = map_data_types(df)

    expected =  {
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
                "thal": "int64",}

    actual = {col:df[col].dtype.name for col in df.columns}
    print(expected)
    print(actual)
    assert expected == actual