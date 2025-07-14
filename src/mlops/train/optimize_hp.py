import os
import mlflow
import pandas as pd
import xgboost as xgb
import optuna
from prefect import task
from sklearn.metrics import roc_auc_score

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("OPTUNA_EXPERIMENT"))

@task
def hyperparameter_search(data_dict:dict, num_trials: int):
    # Define Matrices
    X_train = xgb.DMatrix(data_dict["x_train"], label=data_dict["y_train"], enable_categorical=True)
    X_val = xgb.DMatrix(data_dict["x_valid"], enable_categorical=True)
    # Define Sample input for mlflow logs
    input_sample = data_dict["x_train"].iloc[[0]]
    #mlflow.autolog()
    def objective(trial: optuna.Trial) -> float:
            with mlflow.start_run(nested=True):
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'tree_method': 'hist',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                    'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
                    }
                
                model = xgb.train(
                            params,
                            X_train,
                            num_boost_round=500,
                            evals=[(X_val, 'val')],
                            early_stopping_rounds=20,
                            verbose_eval=False
                        )

                preds = model.predict(X_val)
                auc = roc_auc_score(data_dict["y_valid"], preds)
                mlflow.log_params(params)
                mlflow.log_metric(key="valid_auc_score",
                                  value=auc,)
                mlflow.xgboost.log_model(model, 
                                         name=f"xgb-model-{trial.number}", 
                                         model_format="ubj",
                                         input_example=input_sample,)
                mlflow.set_tag("pipeline", "optuna_hp_search")

                return auc
    # Create study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials, gc_after_trial=True)