import os
from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from plotly.io import to_json
from framework.ExperimentHub import ExperimentHub

import logging

exp_hub : ExperimentHub = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global exp_hub
    ## NOTE: docker run -e MY_USER=test -e MY_PASS=12345 ... <image-name> ..

    # NOTE: mlflow uri = sys.argv[1]    #os.environ['MLFLOW_URI'] = 
    #os.environ['RABBIT_HOST'] = "localhost"
    #os.environ['RABBIT_PORT'] = "5100"
    #os.environ['RABBIT_USER'] = "guest"
    #os.environ['RABBIT_PASSWORD'] = "guest"
    mlflow_uri = os.environ['MLFLOW_URI']  #"http://localhost:5000" 
    rabbit_host =  os.environ['RABBIT_HOST'] #"localhost"
    rabbit_port = os.environ['RABBIT_PORT'] #"5100"
    rabbit_user = os.environ['RABBIT_USER'] # "guest" #
    rabbit_pass = os.environ['RABBIT_PASSWORD'] #"guest" #
    ##TODO: add db to connect and restore previous experiments metadata...
    exp_hub = ExperimentHub(mlflow_uri=mlflow_uri, rabbit_host=rabbit_host, rabbit_port=rabbit_port, rabbit_user=rabbit_user, rabbit_password=rabbit_pass)
    yield

    exp_hub._disconnect_from_rabbit()
    ##TODO: save state of experiment hub---
    
    return ## do nothing

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    """Default path. See /docs for more."""
    return "Hello World"
    ## TODO:

@app.post("/flush/{queue}")
def flush_rabbit(queue : str):
    """Remove all data from rabbit MQ queue {queue}."""
    global exp_hub
    exp_hub._flush(queue)
    return "Rabbit flushed"

@app.post("/{name}/train")
async def train_model(name:str, request: Request, background_tasks: BackgroundTasks):
    """Sets up experiment."""
    global exp_hub
    a = await request.body()
    metadata = json.loads(a.decode('utf-8'))
    #await exp_hub.train(name, metadata)
    background_tasks.add_task(exp_hub.train, name, metadata)
    return {"message" : "Training started based on sent parameters."}


@app.post("/{name}/load")    
def load_experiment(name:str): ## ???
    global exp_hub
    exp_hub.load(name)
    return "Experiment loaded"

@app.post("/{name}/load/{run_id}")
def load_experiment_run_id(name:str, run_id:str):
    """Loads an experiment as {name} in the ExperimentHub based on a {run_id} mlflow run id identifier."""
    global exp_hub
    exp_hub.load(name, run_id)
    return "Experiment loaded"

@app.post("/{name}/save")
def save_experiment(name:str):
    """Saves experiment to MLFLOW server."""
    global exp_hub
    exp_hub.save(name)
    return "Experiment saved"

@app.post("/{name}/predict")
async def predict(name:str, background_tasks: BackgroundTasks):
    """Calls "predict" function of experiment {name}."""
    global exp_hub 
    ##TODO: add celery to ExperimentHub
    ## TODO: add webhook here

    if exp_hub.available.get(name, False):
        background_tasks.add_task(exp_hub.predict, name)
        #await exp_hub.predict(name)
    ## TODO: add metrics db here...
        return "Prediction called"
    else:
        return Response(status_code=204)

@app.get("/{name}/plot/{plot_name}")
def plot(name:str, plot_name:str):
    """Returns the model's figure {plot_name} of experiment {name}."""
    global exp_hub
    
    if name != 'None' and plot_name != 'None':
        return Response(content=to_json(exp_hub.plot(name, plot_name)), status_code=200)
    else:
        return Response(status_code=204)

@app.get("/{name}/plot_eda/{plot_name}")
def plot(name:str, plot_name:str):
    """Returns the EDA figure {plot_name} of experiment {name}."""
    global exp_hub
    if name != 'None' and plot_name != 'None':
        return Response(content=to_json(exp_hub.plot_eda(name, plot_name)), status_code=200)
    else:
        return Response(status_code=204)
    
##maybe integrate this with rabbit to send it to the frontend...
@app.get("/{name}/train_data")
def get_train(name :str):
    """Returns train data.""" ## Indices???
    global exp_hub
    return Response(content = exp_hub.get_train_data(name), status_code=200)

@app.get("/{name}/cfg")
def get_cfg(name:str):
    """Get configuration of an experiment"""
    global exp_hub
    return JSONResponse(content = exp_hub.get_cfg(name), status_code=200)

@app.get("/experiments")
def get_experiments():
    """Returns experiment names and figure names."""
    if not exp_hub.experiments:
        return Response(status_code=204)
    else:
        exps = dict()
        for k in exp_hub.experiments.keys():
            exps[k] = exp_hub.plot_names(k)
        return JSONResponse(content = exps, status_code=200)

####

@app.post("/{name}/kill")
def kill_experiment(name:str):
    """Stops training of a model. NOT WORKING..."""
    global exp_hub
    exp_hub.kill(name)
    return "Experiment killed"


@app.get("/{name}/run_id")
def get_exp_runid(name : str):
    """Returns run ID."""
    global exp_hub
    return exp_hub.experiments[name].run_id

@app.get("/{name}/exp_id")
def get_exp_expid(name : str):
    """Returns experiment ID."""
    global exp_hub
    return exp_hub.experiments[name].experiment_id


@app.get("/{name}/results")
def get_results(name:str):
    "Returns current results of a model."
    # TODO: return predict
    pass

@app.get("/log")
def get_log():
    """Returns log file."""
    pass

@app.post("/{name}/retrain")
def retrain(name : str, background_tasks: BackgroundTasks): ## add background task
    global exp_hub
    background_tasks.add_task(exp_hub.retrain, name)
    return "Retraining started as requested."

### Add request for metric, threshold, change...


if __name__ == "__main__":
    import uvicorn
    os.environ['MLFLOW_URI'] = "http://localhost:5000"
    os.environ['RABBIT_HOST'] = "localhost" 
    os.environ['RABBIT_PORT'] = "5100"
    os.environ['RABBIT_USER'] = "guest"
    os.environ['RABBIT_PASSWORD'] = "guest"
    os.environ['HUB_URL'] = "http://0.0.0.0:8010"
    os.system("uvicorn ExperimentHubAPI:app --host localhost --port 8010")
    #uvicorn.run(app, host="localhost", port="8010", log_level="info")