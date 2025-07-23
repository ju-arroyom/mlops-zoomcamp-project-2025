import asyncio
from datetime import datetime, timezone

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from prefect import get_client
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from prefect.client.schemas.sorting import FlowRunSort

from mlops.inference.predict import load_model, make_prediction
from mlops.processing.prepare_features import map_data_types
from mlops.monitoring.metrics_calculation import calculate_metrics, insert_metrics_to_db

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    latest_run = await get_most_recent_flow_run()
    # Get base url
    base_url = str(request.base_url)
    mlflow_url = base_url.replace(str(request.url.port), "5500")
    streamlit_url = base_url.replace(str(request.url.port), "8501")
    prefect_url = base_url.replace(str(request.url.port), "4200")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "mlflow_url": mlflow_url,
            "streamlit_url": streamlit_url,
            "prefect_url": prefect_url,
            "latest_run": latest_run[0],
        },
    )


async def get_most_recent_flow_run() -> list:
    async with get_client() as client:
        # Read flow runs, sorting by end time in descending order to get the most recent first
        # Limit to 1 to get only the single most recent flow run
        recent_flow_runs = await client.read_flow_runs(
            sort=FlowRunSort.END_TIME_DESC,
            limit=1,
        )
        return recent_flow_runs


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
    result = {"predicted_heart_disease": int(row["prediction"].values)}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
