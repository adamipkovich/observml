"""CLI utility commands for the MMLW API"""
import os
import click
import pika
import time
import pandas as pd
import pickle
import json
import yaml
from yaml import SafeLoader
from api_commands import run_test_project_template, compile_config, train
import webbrowser
## MMLW CLI CONTEXT SETTINGS
CONTEXT_SETTINGS = dict(
    default_map={'runserver': {'port': 5000}}
)

#### DEVELOPMENT COMMANDS
#### API COMMUNICATION/FEATURE COMMANDS


@click.group("prod")
def prod():
    """Development commands for the MMLW API, e.g. locally starting required services."""
    pass

@prod.command("mlflow")
def start_mlflow():
    """Opens mlflow client."""
    webbrowser.open("http://localhost:5000")


@prod.command("pull-images")
def pull ():
    os.system("docker pull rabbitmq:3.12.14-management-alpine")
    ## pull custom mlflow
    ## pull backend
    ## pull frontend

@click.option("--host", default = "localhost", help="RabbitMQ Host")
@click.option("--port", default = "15100", help="RabbitMQ inspection TCP port")
@prod.command("rabbitmq")
def open_rabbitmq(port, host):
    webbrowser.open(f"http://{host}:{port}")


@click.option("--port", default = "8010", help="Port to run the backend server on")
@click.option("--host", default = "localhost",  help="Host to run the backend server on")
@prod.command("backend")
def open_backend(port, host):
    """Starts the backend server on the specified host and port, with the specified RabbitMQ connection and MLFlow connection."""
    webbrowser.open(f"http://{host}:{port}/docs")    


@click.option("--port", default = "8105", help="Port to run the frontend server on")
@click.option("--host", default = "localhost",  help="Host to run the frontend server on")
@prod.command("frontend")
def start_frontend(port, host):
    """Starts the frontend server on the specified host and port."""
    webbrowser.open(f"http://{host}:{port}")
    
    #os.environ['HUB_URL'] = hub_url
    #os.system("streamlit run streamlit_frontend.py")


@click.option("--hub_url", default = "http://localhost:8010", help="URL of the Experiment Hub (backend)")
@click.option("--rabbit_host", default = "localhost", help="RabbitMQ host")
@click.option("--rabbit_port", default = "5672", help="RabbitMQ port")
@click.argument("script", default = "train_script_obj.py")
@prod.command("test")
def test_app(script):
    """Runs the test suite for the Experiment Hub API.
    Args:
        script: Path to the test script. e.g. test_script_obj.py must be self-contained.
    Options:
        --hub_url: URL of the Experiment Hub (backend)
        --rabbit_host: RabbitMQ host
        --rabbit_port: RabbitMQ port    
        """
    os.system(f"python {script}")


@prod.command("make-project")
def make_project():
    """Build hydra project for backend configuration. Copies example project to current directory from git."""
    os.system("git clone ")
    

@click.command("run")
def run():
    ## check if images are built
    os.system("docker compose up")

if __name__ == "__main__":
    prod()



#_______________________________________________________________________________________________________________________
# ## Add context to the command - store the configuration file in the context, store rabbit connection in the context?
# @click.option("--config_name", default = "config", help="Name of the configuration file")
# @click.option("--config_path", default = "./configs", help="Path to create the project in")
# @click.option("--rabbit_host", default = "localhost", help="RabbitMQ host")
# @click.option("--rabbit_port", default = "5672", help="RabbitMQ port")
# @click.option("--rabbit_user", default = "guest", help="RabbitMQ user")
# @click.option("--rabbit_password", default = "guest", help="RabbitMQ password")
# @click.option("--hub_url", default = "http://localhost:8100", help="URL of the Experiment Hub (backend)")
# @click.argument("data_path", default = "data.yaml")

# @dev.command("train")
# def train_model(data_path, config_name, config_path, rabbit_host, rabbit_port, rabbit_user, rabbit_password, hub_url):
#     """ COMMAND: data_path model_name data_tag --config_name --config_path --rabbit_host --rabbit_port --rabbit_user --rabbit_password --hub_url
#     data_path: Path to the data files to be used for training. Must be in yaml format, with the key as the model name (rabbit-queue) and the value as the path to the data file.
#     Description:
#         Trains a model using the specified configuration, with a specific training set on the disk. To use it from memory, one should use api_commands.py in a script. 
#         Data must be given in an xlsx/csv/json/txt format. Accepts pickled pandas Dataframe files. 
#         Data must be named as the queue in rabbitmq (key in data_path (e.g. detect key in data.yaml)).
#         There is an example for script-based data posting for training (train_script.py).
#         """
    
#     cfg = compile_config(config_path, config_name)  
#     with open(data_path, 'r') as f:
#         data_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
#     for k in cfg.keys():
#         ## read data - xlsx, csv, json, pickle
#         if data_cfg[k].endswith(".xlsx"): #endswith
#                 X = pd.read_excel(data_cfg[k])
#                 X = X.to_json()
#         elif data_cfg[k].endswith(".csv"):
#                 X = pd.read_csv(data_cfg[k])
#                 X = X.to_json()
#         elif data_cfg[k].endswith(".txt"):
#                 X = pd.read_csv(data_cfg[k])
#                 X = X.to_json()
#         elif data_cfg[k].endswith(".json"):
#                 X = pd.read_json(data_cfg[k])
#         elif data_cfg[k].endswith(".pkl"):
#                 with open(data_cfg[k], "rb") as f:
#                     X = pickle.load(f)
#                     X = pd.DataFrame(X).to_json()
#         else:   
#             click.echo(f"Data file {data_path[k]} does not exist, or cannot load. Please ensure the file is in the correct format (xlsx, csv, txt, json, pkl).")
#             return

#         train(cfg[k], url=hub_url, model_name = k, data = X, rabbit_host = rabbit_host, rabbit_port = rabbit_port, user = rabbit_user, password = rabbit_password)

# @click.option("--port", default = 5672, help="RabbitMQ port")
# @click.option("--password", default = "guest",  help="RabbitMQ password")
# @click.option("--rabbit_user", default = "guest",  help="RabbitMQ user")
# @click.option("--tag", default = "latest", help="RabbitMQ image tag")
# @dev.command("rabbitmq")
# def start_rabbitmq(port, password, rabbit_user, tag):
#         """Checks if RabbitMQ server is running, if not, starts it."""
#         ## check if rabbit image exists, pull if not.
#         #click.echo(os.system(f"docker manifest inspect rabbitmq:{tag}"))
#         #os.system(f"docker pull rabbitmq:{tag}")

#         try: 
#             click.echo("Checking if RabbitMQ server is running...")
#             connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', port=port, credentials=pika.PlainCredentials(rabbit_user, password )))
#             connection.close()
#             click.echo(f"CLI Response: RabbitMQ server already running on http://localhost:{port}")
#         except pika.exceptions.AMQPConnectionError:
#             click.echo("CLI Response: RabbitMQ server not running.")
#             os.system(f"docker run -d -p {port}:5672 rabbitmq")
#             try:
#                 connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', port=port, credentials=pika.PlainCredentials(rabbit_user, password )) )
#                 click.echo("Please wait for RabbitMQ server to start...")
#                 time.sleep(4)
#                 click.echo("CLI Response: RabbitMQ server started on http://localhost:5672")
#                 connection.close()
#             except pika.exceptions.AMQPConnectionError:
#                 click.echo("CLI Response: RabbitMQ server failed to start.")
#                 click.echo("Attempting to start RabbitMQ server on Docker...")
#                 os.system(f"docker run -d -p {port}:5672 rabbitmq:{tag}")

