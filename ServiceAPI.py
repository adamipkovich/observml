"""Microservice for training models. Also contains retrain functionality."""
import logging
from fastapi import FastAPI, Request
from fastapi.responses import Response
import pandas as pd
import json
import mlflow
import os
from tools import load_object
import pika
import yaml
from yaml import SafeLoader
from contextlib import asynccontextmanager
from plotly.io import to_json

from mlflow.client import MlflowClient

## TODO: pull code from github/gitlab/VC system!!!! importlib.reload,
## TODO: Celery worker...
## TODO: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads https://www.pgadmin.org/download/pgadmin-4-windows/
## TODO: Add email sending if retrained/problem...
#mod_interface : ModelServerInterface = ModelServerInterface()
experiment = None
data_stream_connection = None
channel = None
chqueue = None
config : dict = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    ## build rabbit mq connection ... 

    yield

    await flush()
    if channel is not None:
        if channel.is_open:
            channel.close()

    if data_stream_connection is not None:
        if data_stream_connection.is_open:
           data_stream_connection.close()

app = FastAPI(lifespan=lifespan) 

def connect_to_rabbit(host:str, queue:str, port:int):
    global data_stream_connection, channel, chqueue
    if data_stream_connection is not None:
        data_stream_connection.close()
        data_stream_connection = None
    data_stream_connection = pika.BlockingConnection(pika.ConnectionParameters(host)) #port=port
    chqueue = queue

    if channel is not None:
        channel.close()
        channel = None

    channel = data_stream_connection.channel()
    channel.queue_declare(queue=queue, durable = True) # durable = True
    channel.basic_qos(prefetch_count=1)


async def get_data(queue: str):
    global data_stream_connection, channel

    if data_stream_connection is None:
        return Response("RabbitMQ Connection is None during obtaining data.")
    elif data_stream_connection.is_closed:
        return Response("The configured Rabbit Connection is closed.")

    if channel is None:
        return Response("The Service has not yet openned a RabbitMQ Channel")
    elif channel.is_closed:
        return Response("The configured RabbitMQ Channel is closed.")

    ## Pull one

    method_frame, header_frame, body = channel.basic_get(queue)
    if method_frame:
        data = pd.read_json(body.decode("utf-8"))
        channel.basic_ack(method_frame.delivery_tag)
        return data        
    else:
        return pd.DataFrame([])

async def train_model(metadata : dict, data, retrain = False):
    if data is None:
        return Response("No data found in queue. Please try again once the data is sent.")

    experiment_class = load_object(metadata["train"]["interface"]["module"], metadata["train"]["interface"]["name"])
    experiment = experiment_class(cfg = metadata["train"])

    mlflow.set_tracking_uri(metadata["deployment"]["tracking_uri"])
    experiment_name = metadata["deployment"]["experiment_name"]
    client = MlflowClient(metadata["deployment"]["tracking_uri"])
    try:
         client.create_experiment(experiment_name)
    except Exception as e:
         pass    
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_tracking_uri(metadata["deployment"]["tracking_uri"])
    with mlflow.start_run(experiment_id=experiment_id):
        #experiment_id = mlflow.active_run().info.experiment_id
        run_id = mlflow.active_run().info.run_id

        if not os.path.exists(os.path.join(os.getcwd(), "runs")):
            os.makedirs(os.path.join(os.getcwd(), "runs"))
        if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id)):
            os.makedirs(os.path.join(os.getcwd(), "runs", run_id))
        if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id, "reports")):
            os.makedirs(os.path.join(os.getcwd(), "runs", run_id, "reports"))
        if not os.path.exists(os.path.join(os.getcwd(), "runs",run_id, "model")):
            os.makedirs(os.path.join(os.getcwd(), "runs", run_id, "model"))
        
        yaml.dump(metadata, open(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml"), "w"))  
        mlflow.log_artifact(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml"), "")
        experiment.run(data, experiment_id, run_id)
        experiment.save()
        
    return experiment


async def predict():
    df = await get_data(chqueue)
    res = experiment.predict(df)
    return res


@app.get("/")
async def read_root():
    return config


@app.post("/train")
async def fit_model(train_request: Request): #,  bcgTask : BackgroundTasks
    """Fit a model to the data"""

    global config, experiment
    a = await train_request.body()
    metadata = json.loads(a.decode('utf-8'))

    ## Connect to rabbit    
    connect_to_rabbit(metadata["data"]["rabbit_connection"], metadata["data"]["queue"], metadata["data"]["port"])

    data = await get_data(chqueue)
    experiment = await train_model(metadata, data)
    config = metadata
    return f"Model trained successfully according to request."

    
@app.get("/load/{run_id}/{mlflow_host}/{mlflow_port}")
async def load_interface(run_id: str, mlflow_host : str, mlflow_port : str):
    """Load the model from the mlflow tracking server"""

    global experiment, config
    mlflow.set_tracking_uri("http://" + mlflow_host + ":" + mlflow_port)
    mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/metadata.yaml", dst_path=os.getcwd())
    
    with open("./metadata.yaml") as f:
        metadata = yaml.load(f, Loader=SafeLoader) 

    interface_class = load_object(metadata["train"]["interface"]["module"], metadata["train"]["interface"]["name"])
    experiment = interface_class(cfg = metadata["train"])
    experiment = experiment.load(run_id=run_id)
    connect_to_rabbit(metadata["data"]["rabbit_connection"], metadata["data"]["queue"], metadata["data"]["port"])
    config = metadata
        
    return {"message": f"Model loaded from mlflow successfully. Serving with id {run_id}."}


@app.post("/flush/rabbit")
async def flush():
    channel.queue_purge(config["data"]["queue"])
    return Response("Channel flushed successfully")


@app.post("/predict")
async def predict_with_rabbit():
    y, m = await predict()
    return Response(content = y.to_json(), status_code=200)
   
@app.get("/experiment/metrics")
async def get_metrics():
    return Response(content=experiment.metrics.to_json()) 

@app.get("/experiment/reports")
async def get_figs():
    fig = json.dumps(experiment._report_registry)
    return Response(content=fig, status_code=200)

@app.get("/experiment/reports/{any_object}")
async def get_figs(any_object):
    fig = experiment._report_registry[any_object]
    return Response(content=to_json(fig), status_code=200)

@app.get("/experiment/new_data")
async def get_predictions():
    return Response(content=experiment.new_data.to_json())

@app.get("/experiment/new_data/{col}")
async def get_predictions_col(col: str):
    return Response(content=experiment.new_data[col].to_json())

@app.get("/experiment/figs")
async def get_figs():
    ser = json.dumps(experiment.get_fig_types())
    return Response(content=ser)

if __name__ == "__main__":
    import sys
    import uvicorn
    ## Deployments should be built with docker-compose - one can parameterize the ports and hosts and the other services.
    rabbit_mq = sys.argv[1] ## rabbit mq connection -- build connection...
    mlflow_uri = sys.argv[2] ## mlflow uri
    chqueue = sys.argv[3] ## queue name
    ## add port??
    uvicorn.run(app, port=8010)
