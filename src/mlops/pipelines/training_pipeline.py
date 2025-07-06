import os
import mlflow
import click
import pandas as pd
import xgboost as xgb
from pathlib import Path
from prefect import flow, get_run_logger, task

from mlops.train.preprocess import Preprocessor
from mlops.train.optimize_hp import hyperparameter_search
from mlops.train.register_model import register_model_to_mlflow


@task
def ingest_data():
    """Ingest data from a source"""
    input_dir =  Path(__file__).parent.parent / "data"
    file_path = input_dir / "cardiac_arrest_dataset.csv"
    heart_df = pd.read_csv(file_path)
    return heart_df


@task
def preprocess_data(data:pd.DataFrame, logger):
    preprocess_task = Preprocessor(data=data, target="target", logger=logger)
    preprocess_task.build_datasets()
    return preprocess_task


@flow
def train_heart_disease_classifier(num_trials: int, top_n:int):
    """
    Train Heart Disease Classifier
    """
    logger = get_run_logger()
    logger.info("✅ Ingesting data...")
    data = ingest_data()
    logger.info("🛠️ Preprocessing data...")
    preprocess_task = preprocess_data(data=data, logger=logger)
    logger.info("🎯 Run Hyperparameter Search")
    hyperparameter_search(data_dict=preprocess_task.data_dict, num_trials=num_trials)
    logger.info("🏆 Retrain Model with Best Params")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500"))
    mlflow.set_experiment(os.getenv("EXPERIMENT_NAME", "xgb_best_model"))   
    register_model_to_mlflow(data_dict=preprocess_task, top_n=top_n)
    logger.info("🚀  Completed Training Pipeline Successfully")


@click.command()
@click.option(
    "--num_trials",
    default=20,
    help="Number of trials for Hyperparameter Search"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def main(num_trials, top_n):
    train_heart_disease_classifier(num_trials=num_trials, top_n=top_n)

if __name__ == "__main__":
    main()