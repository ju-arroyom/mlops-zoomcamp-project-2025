import os
from pathlib import Path

import click
import mlflow
import pandas as pd
from prefect import flow, task, get_run_logger

from mlops.train.optimize_hp import hyperparameter_search
from mlops.train.register_model import register_model_to_mlflow
from mlops.processing.preprocess import Preprocessor


@task
def ingest_data():
    """Ingest data from a source"""
    storage_type = os.getenv("STORAGE_TYPE", "local")
    if storage_type == "s3":
        bucket = os.getenv("S3_BUCKET", "my-bucket")
        file_path = f"s3://{bucket}/data/cardiac_arrest_dataset.csv"
        storage_options = {"client_kwargs": {"endpoint_url": "http://localhost:4566"}}
        return pd.read_csv(file_path, storage_options=storage_options)
    else:
        input_dir = Path(__file__).parent.parent / "data"
        file_path = input_dir / "cardiac_arrest_dataset.csv"
        return pd.read_csv(file_path)


@task
def preprocess_data(data: pd.DataFrame):
    """
    Run Preprocessor class

    Args:
        data (pd.DataFrame): df to process

    Returns:
        Dataframe: Processed dataframe
    """
    preprocess_task = Preprocessor(data=data, target="target")
    preprocess_task.build_datasets()
    return preprocess_task


@task
def write_data(data: pd.DataFrame, name: str):
    """
    Write datasets to folder

    Args:
        data (pd.DataFrame): df to write
        name (str): name of the dataframe (train/test)
    """
    storage_type = os.getenv("STORAGE_TYPE", "local")
    if storage_type == "s3":
        bucket = os.getenv("S3_BUCKET", "my-bucket")
        file_path = f"s3://{bucket}/data/{name}_dataset.parquet"
    else:
        output_dir = Path(__file__).parent.parent / "data"
        file_path = output_dir / f"{name}_dataset.parquet"
    try:
        data.to_parquet(
            file_path,
            storage_options=(
                {"client_kwargs": {"endpoint_url": "http://localhost:4566"}}
                if storage_type == "s3"
                else None
            ),
        )
        print(f"Writing Data to path: {file_path}")
    except OSError as e:
        print(f"❌ File write error: {e}")


@flow
def train_heart_disease_classifier(num_trials: int, top_n: int):
    """
    Train Heart Disease Classifier
    """
    logger = get_run_logger()
    prefect_api_url = os.getenv("PREFECT_API_URL")
    logger.info(f"Prefect API URL {prefect_api_url}")
    logger.info("✅ Ingesting data...")
    data = ingest_data()
    logger.info("🛠️ Preprocessing data...")
    preprocess_task = preprocess_data(data=data)
    logger.info("🎯 Run Hyperparameter Search")
    hyperparameter_search(data_dict=preprocess_task.data_dict, num_trials=num_trials)
    logger.info("✅ Write Datasets ...")
    write_data(preprocess_task.full_df, "train")
    write_data(preprocess_task.df_test, "test")
    logger.info("🏆 Retrain Model with Best Params")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500"))
    mlflow.set_experiment(os.getenv("EXPERIMENT_NAME", "xgb_best_model"))
    register_model_to_mlflow(data_dict=preprocess_task, top_n=top_n)
    logger.info("🚀  Completed Training Pipeline Successfully")


@click.command()
@click.option(
    "--num_trials", default=20, help="Number of trials for Hyperparameter Search"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote",
)
def main(num_trials, top_n):
    """Main function to run pipeline"""
    train_heart_disease_classifier(num_trials=num_trials, top_n=top_n)


if __name__ == "__main__":
    main()
