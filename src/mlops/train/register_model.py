import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
from prefect import task
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score




def retrain_and_test_models(data_dict: dict, params: dict):
    """_summary_

    Args:
        data_dict (dict): _description_
        params (dict): _description_
    """
    df_train = data_dict.full_df.drop("target", axis=1)
    y_train = data_dict.full_df['target'].values
    mlflow.xgboost.autolog()
    with mlflow.start_run():
        X_train = xgb.DMatrix(df_train, label=y_train, enable_categorical=True)
        model = xgb.train(dtrain=X_train, params=params)
        # Prepare Test Dataframe
        df_test = data_dict.df_test.drop("target", axis=1)
        y_test = data_dict.df_test['target'].values
        X_test = xgb.DMatrix(df_test,enable_categorical=True)
        # Predict
        predictions = model.predict(X_test)
        auc = roc_auc_score(y_test, predictions)
        mlflow.log_metric("test_auc_score", auc)



@task
def register_model_to_mlflow(data_dict: dict, top_n: int):
    """_summary_

    Args:
        data_dict (dict): _description_
        top_n (int): _description_
    """
    client = MlflowClient()
    # Retrieve the top_n model runs and log the models
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "xgb_optuna_search")
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.valid_auc_score DESC"]
    )
    for run in runs:
        retrain_and_test_models(data_dict=data_dict, params=run.data.params)

    # Select the model with the lowest test RMSE
    final_experiment_name = os.getenv("EXPERIMENT_NAME", "xgb_best_model")
    experiment = client.get_experiment_by_name(final_experiment_name)
    best_run = client.search_runs(experiment_ids=experiment.experiment_id,
                                 run_view_type=ViewType.ACTIVE_ONLY,
                                  order_by=["metrics.test_auc_score DESC"])[0]

    print(f"run: {best_run.info.run_id}, test rmse: {best_run.data.metrics['test_auc_score']:.4f}")
    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=final_experiment_name)
