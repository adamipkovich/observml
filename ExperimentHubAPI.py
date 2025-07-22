import os
from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import Response, JSONResponse
from plotly.io import to_json
from framework.ExperimentHub import ExperimentHub

import logging

exp_hub : ExperimentHub = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global exp_hub
    
    # Check if config file exists
    config_path = os.environ.get('HUB_CONFIG_PATH', 'hub_config.yaml')
    if os.path.exists(config_path):
        # Create ExperimentHub from configuration
        exp_hub = ExperimentHub.from_config(config_path)
        logging.info(f"ExperimentHub initialized from config file: {config_path}")
    else:
        # Fall back to environment variables
        logging.info(f"Config file {config_path} not found, using environment variables")
        mlflow_uri = os.environ.get('MLFLOW_URI', "http://localhost:5000")
        rabbit_host = os.environ.get('RABBIT_HOST', "localhost")
        rabbit_port = os.environ.get('RABBIT_PORT', "5672")
        rabbit_user = os.environ.get('RABBIT_USER', "guest")
        rabbit_pass = os.environ.get('RABBIT_PASSWORD', "guest")
        
        exp_hub = ExperimentHub(
            mlflow_uri=mlflow_uri, 
            rabbit_host=rabbit_host, 
            rabbit_port=rabbit_port, 
            rabbit_user=rabbit_user, 
            rabbit_password=rabbit_pass
        )
    
    yield
    
    # TaskQueue plugin removed - no longer supported
    pass
    
    # Shutdown plugins
    for plugin in exp_hub.plugins.values():
        plugin.shutdown()

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
    exp_hub.flush(queue)
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
def plot_eda(name:str, plot_name:str):
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

@app.post("/{name}/stop_training")
def stop_training(name:str):
    """Stops the training process for an experiment without removing it."""
    global exp_hub
    success = exp_hub.stop_training(name)
    if success:
        return {"status": "success", "message": f"Training for experiment {name} stopped successfully"}
    else:
        return {"status": "error", "message": f"Failed to stop training for experiment {name}"}

@app.post("/{name}/delete")
def delete_experiment(name:str):
    """Removes an experiment from memory."""
    global exp_hub
    success = exp_hub.delete_experiment(name)
    if success:
        return {"status": "success", "message": f"Experiment {name} deleted successfully"}
    else:
        return {"status": "error", "message": f"Failed to delete experiment {name}"}

@app.post("/{name}/kill")
def kill_experiment(name:str):
    """Legacy endpoint. Stops training of a model. Use /stop_training instead."""
    global exp_hub
    success = exp_hub.stop_training(name)
    if success:
        return {"status": "success", "message": f"Experiment {name} training stopped successfully"}
    else:
        return {"status": "error", "message": f"Failed to stop experiment {name} training"}

@app.post("/{name}/kill/{task_id}")
def kill_specific_task(name:str, task_id:str):
    """Legacy endpoint - task queue functionality removed."""
    return {"status": "error", "message": "Task queue functionality has been removed. Use /stop_training instead."}


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

@app.get("/health")
def check_health():
    """Check the health of all plugins"""
    global exp_hub
    return JSONResponse(content=exp_hub.check_plugin_health())

@app.get("/available_experiments")
def get_available_experiment_types():
    """Get available experiment types"""
    global exp_hub
    return JSONResponse(content=exp_hub.available_experiments)

@app.post("/create_experiment/{name}/{experiment_type}")
async def create_experiment(name: str, experiment_type: str, request: Request, background_tasks: BackgroundTasks):
    """Create a new experiment of the specified type"""
    global exp_hub
    
    # Check if experiment type is available
    if experiment_type not in exp_hub.available_experiments:
        raise HTTPException(status_code=400, detail=f"Unknown experiment type: {experiment_type}")
    
    # Get configuration from request body
    a = await request.body()
    config = json.loads(a.decode('utf-8'))
    
    # Create experiment
    try:
        background_tasks.add_task(exp_hub.create_experiment, name, experiment_type, config)
        return {"message": f"Creating experiment '{name}' of type '{experiment_type}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
