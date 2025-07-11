REQUIRED_TYPES = {
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

def map_data_types(df):
    df = df.astype(REQUIRED_TYPES)
    return df
