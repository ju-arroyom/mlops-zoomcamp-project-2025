import asyncio
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
#from fastapi.responses import HTMLResponse, PlainTextResponse
from datetime import datetime, timedelta, timezone
import pandas as pd
from mlops.inference.predict import load_model, make_prediction
from mlops.processing.prepare_features import map_data_types
from mlops.monitoring.metrics_calculation import calculate_metrics, insert_metrics_to_db

app = FastAPI()

async def delayed_insert(metrics):
    await asyncio.sleep(5)
    insert_ts = datetime.now(timezone.utc)
    insert_metrics_to_db(metrics, insert_ts)

@app.post("/predict")
async def predict(request: Request, background_tasks: BackgroundTasks):
        # Convert input to DataFrame
        data_dict = await request.json()
        row = pd.DataFrame.from_dict(data_dict)
        row = map_data_types(row)
        # load model
        model = load_model()
        row = make_prediction(model, row)
        import evidently
        print("FastAPI Env:", evidently.__version__)
        metrics = calculate_metrics(row)
        # schedule the delayed insert (with fresh timestamp inside)
        background_tasks.add_task(delayed_insert, metrics)
        result = {"predicted_heart_disease": int(row['prediction'].values)}
        return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
      