import os
from pathlib import Path

import pandas as pd
import sqlalchemy
from evidently import Report, Dataset, DataDefinition
from evidently.metrics import (
    ValueDrift,
    MissingValueCount,
    DriftedColumnsCount
)

from mlops.processing.prepare_features import map_data_types


def read_reference_dataset():
    """
    Read reference dataset

    Returns:
        Dataframe: Pandas df for reference data
    """
    input_dir = Path(__file__).parent.parent / "data"
    path = input_dir / "train_dataset.parquet"
    df = pd.read_parquet(path)
    df.rename(columns={"target": "prediction"}, inplace=True)
    df["prediction"] = df["prediction"].astype(int)
    return map_data_types(df)


def create_table():
    """
    Create Database for prediction metrics

    Returns:
        tuple: Tuple with connection engine and table object
    """
    engine = sqlalchemy.create_engine(
        os.getenv(
            "METRICS_DB_URI", "postgresql://user:pass@localhost:5432/prediction_metrics"
        )
    )
    metadata = sqlalchemy.MetaData()
    metrics_table = sqlalchemy.Table(
        "prediction_metrics",
        metadata,
        sqlalchemy.Column("timestamp", sqlalchemy.DateTime),
        sqlalchemy.Column("prediction_drift", sqlalchemy.Float),
        sqlalchemy.Column("drifted_columns", sqlalchemy.Integer),
        sqlalchemy.Column("missing_value_share", sqlalchemy.Float),
    )
    metadata.create_all(engine)
    return engine, metrics_table


def calculate_metrics(current_data):
    """
    Calculate drift metrics for prediction

    Args:
        current_data (pd.Dataframe): Single row to predcit

    Returns:
        dict: Dictionary with metrics results
    """
    try:
        num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

        data_definition = DataDefinition(
            numerical_columns=num_features + ["prediction"],
            categorical_columns=cat_features,
        )

        report = Report(
            metrics=[
                ValueDrift(column="prediction"),
                DriftedColumnsCount(),
                MissingValueCount(column="prediction"),
            ]
        )

        reference_data = read_reference_dataset()
        # Single rows cause error
        current_data = pd.concat([current_data, current_data], ignore_index=True)
        print(current_data)
        print("Creating current_dataset...")
        current_dataset = Dataset.from_pandas(
            current_data, data_definition=data_definition
        )

        print("Creating reference_dataset...")
        reference_dataset = Dataset.from_pandas(
            reference_data, data_definition=data_definition
        )

        print("Running report...")
        run = report.run(reference_data=reference_dataset, current_data=current_dataset)
        result = run.dict()

        output = {
            "prediction_drift": float(result["metrics"][0]["value"]),
            "drifted_columns": int(result["metrics"][1]["value"]["count"]),
            "missing_value_share": float(result["metrics"][2]["value"]["share"]),
        }

        return output

    except Exception as e:
        print("Error in calculate_metrics:", e)
        raise


def insert_metrics_to_db(metrics: dict, timestamp):
    """
    Insert metrics to db

    Args:
        metrics (dict): Dictionary with metrics from prediction
        timestamp (timestamp): Timestamp for prediction
    """
    engine, metrics_table = create_table()
    metrics["timestamp"] = timestamp
    with engine.begin() as conn:
        conn.execute(metrics_table.insert().values(**metrics))
