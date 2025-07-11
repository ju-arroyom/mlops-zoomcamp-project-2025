import os
import sqlalchemy
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from evidently import Report
from evidently import DataDefinition
from evidently import Dataset
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount





# GLOBAL VARS
CONNECTION_STRING = "host=localhost port=5432 user=postgres password=example"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=test"

def read_reference_dataset():
    input_dir =  Path(__file__).parent.parent / "data"
    path  = input_dir / "train_dataset.parquet"
    return pd.read_parquet(path)

begin = datetime.now()
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
reference_data = read_reference_dataset()
reference_data.rename(columns={"target":"prediction"}, inplace=True)

data_definition = DataDefinition(
    numerical_columns=num_features + ['prediction'],
    categorical_columns=cat_features,
)

report = Report(metrics = [
    ValueDrift(column='prediction'),
    DriftedColumnsCount(),
    MissingValueCount(column='prediction'),
])


def create_table():
	engine = sqlalchemy.create_engine(os.getenv("METRICS_DB_URI", 
											 "postgresql://user:pass@localhost:5432/prediction_metrics"))
	metadata = sqlalchemy.MetaData()
	metrics_table = sqlalchemy.Table("prediction_metrics", metadata,
        sqlalchemy.Column("timestamp", sqlalchemy.DateTime),
		sqlalchemy.Column("prediction_drift", sqlalchemy.Float),
        sqlalchemy.Column("drifted_columns", sqlalchemy.Integer),
        sqlalchemy.Column("missing_value_share", sqlalchemy.Float),
        )
	metadata.create_all(engine)
	return engine, metrics_table


def calculate_metrics(current_data):
	current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)
	reference_dataset = Dataset.from_pandas(reference_data, data_definition=data_definition)
	run = report.run(reference_data=reference_dataset, current_data=current_dataset)
	result = run.dict()
	prediction_drift = result['metrics'][0]['value']
	num_drifted_columns = result['metrics'][1]['value']['count']
	share_missing_values = result['metrics'][2]['value']['share']
	output = {"prediction_drift": float(prediction_drift),
		      "drifted_columns": int(num_drifted_columns),
			  "missing_value_share": float(share_missing_values)}
	return output



def insert_metrics_to_db(metrics: dict):
	engine, metrics_table = create_table()
	metrics["timestamp"] = datetime.now(timezone.utc) 
	with engine.begin() as conn:
		conn.execute(metrics_table.insert().values(**metrics))
