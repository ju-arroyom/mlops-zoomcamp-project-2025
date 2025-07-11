from fastapi import FastAPI, Request
#from fastapi.responses import HTMLResponse, PlainTextResponse

import pandas as pd
from mlops.inference.predict import load_model, make_prediction
from mlops.inference.prepare_features import map_data_types
from mlops.monitoring.metrics_calculation import calculate_metrics, insert_metrics_to_db
from io import BytesIO

app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict(request: Request):
        # Convert input to DataFrame
        try:
            data_dict = await request.json()
            row = pd.DataFrame.from_dict(data_dict)
            row = map_data_types(row)
            row = make_prediction(model, row)
            #metrics = calculate_metrics(row)
            #insert_metrics_to_db(metrics)
            result = {"predicted_calls": int(row['prediction'])}
            return result
        except Exception as e:
             print("❌ Error during prediction:", e)
             return {"error": str(e)}