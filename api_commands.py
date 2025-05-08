import time
from hydra import initialize, compose
from omegaconf import OmegaConf
import yaml
from yaml import SafeLoader
import json
import requests
import webbrowser
import pika
import logging
import pandas as pd
import os
import asyncio
import pickle
#from scipy.stats.qmc import LatinHypercube
def retrain( name : str, url = "http://localhost:8010"):
     response = requests.post(url=url+ f"/{name}" + "/retrain")

########### API COMMUNICATION/FEATURE COMMANDS
def train(cfg:dict, url="http://localhost:8010", model_name = "model", data = None, rabbit_host = "localhost", rabbit_port = "5100", user = "guest", password = "guest"):
    """Sends only one training request to the backend.
    
    Parameters:
            cfg (dict): Configuration dictionary.
            url (str): URL of the backend.
            model_name (str): Name of the model.
            data (str): Data in JSON format. (must be converted to json, if not json)
            rabbit_host (str): RabbitMQ host.
            rabbit_port (str): RabbitMQ port.
            user (str): RabbitMQ user.
            password (str): RabbitMQ password.

    Returns:
        response: Response from the backend.

    """
    post_data(data, url=url, queue_name = model_name, host = rabbit_host, port = rabbit_port, user = user, password = password)
    json_object = json.dumps(cfg, indent=4)
    response = requests.post(url=url+ f"/{model_name}" + "/train", data=json_object)
    return response

def train_models(conf_path:str = "./configs",
                conf_name:str = "config",
                data_path:str = "data.yaml", 
                url:str="http://localhost:8010",
                rabbit_host:str = "localhost",
                rabbit_port:str = "5100", 
                user:str = "guest",
                password:str = "guest"):
    """Sends project template and related data to the backend. For each key in the config, it will initiate a training request and post the relevant data to the relevant queue.
    
    Parameters: 
            conf_path (str): Path to the configuration files.
            conf_name (str): Name of the configuration file.
            data_path (str): Path to the data file. Requires the same keys as the configuration file.
            url (str): URL of the backend.
            rabbit_host (str): RabbitMQ host.
            rabbit_port (str): RabbitMQ port.
            user (str): RabbitMQ user.
            password (str): RabbitMQ password.

    Returns:
        response: Response from the backend.
    
    """
    
    cfg = compile_config(conf_path, conf_name)  
    with open(data_path, 'r') as f:
        data_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    for k in cfg.keys():
        
        ## read data - xlsx, csv, json, pickle
        if data_cfg[k].endswith(".xlsx"): #endswith
                X = pd.read_excel(data_cfg[k])
                X = X.to_json()
        elif data_cfg[k].endswith(".csv"):
                X = pd.read_csv(data_cfg[k])
                X = X.to_json()
        elif data_cfg[k].endswith(".txt"):
                X = pd.read_csv(data_cfg[k])
                X = X.to_json()
        elif data_cfg[k].endswith(".json"):
                
            with open(data_cfg[k], 'r') as f:
                X = json.dumps((json.load(f)))
                
        elif data_cfg[k].endswith(".pkl"):
                with open(data_cfg[k], "rb") as f:
                    X = pickle.load(f)
                    X = pd.DataFrame(X).to_json()
        else:   
            print(f"Data file {data_path[k]} does not exist, or cannot load. Please ensure the file is in the correct format (xlsx, csv, txt, json, pkl).")
            return
        flush_rabbit(url, k)
        rep = train(cfg[k], url=url, model_name = k, data = X, rabbit_host = rabbit_host, rabbit_port = rabbit_port, user = user, password = password)
        print(f"Response: {rep.content}")


def load_experiment(address : str = "http://localhost:8010", model_name = "model", run_id:str = None):
    """Sends a load request to the backend.

    Parameters:
            address (str): URL of the backend.
            model_name (str): Name of the model.
            run_id (str): Run ID of the model.  
            
    Returns:
        response: Response from the backend.
    """

    if run_id is None:
        resp = requests.post(f"{address}/{model_name}/load")
    else:
        resp = requests.post(f"{address}/{model_name}/load/{run_id}")
    return resp.text

def save_experiment(address : str = "http://localhost:8010", model_name = "model"):
    """Sends a save request to the backend.

    Parameters:
            address (str): URL of the backend.
            model_name (str): Name of the model.
            
    Returns:
        response: Response from the backend.
    """
    resp = requests.post(f"{address}/{model_name}/save")
    return resp.text

def get_monitoring_page(address : str = "http://localhost:8010", tag : str = None, open = True):
     
     if open:
        webbrowser.open(f"{address}/monitor/{tag}")
     
     else :
         rq = requests.get(f"{address}/monitor/{tag}")
         return rq.text
         

def post_data(data, url = "http://localhost:8010", queue_name = "detect_dataset", host = "localhost", port = 5672, user = "guest", password = "guest"):
    """Posts data to the RabbitMQ server. I do not recommend using this function directly, as it can cause issues with data pulling from the queues."""
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, credentials=pika.PlainCredentials(user, password)))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_publish(exchange='', routing_key=queue_name, body=data.encode('utf-8'))
    connection.close()

  
def predict(url = "http://localhost:8010", data = None, model_name = "model", rhost = "localhost", rport = 5672, user = "guest", password = "guest"):
    """Sends a prediction request to the backend. Post data to RabbitMQ.
    
    Parameters:
            url (str): URL of the backend.
            data (str): Data in JSON format. (must be converted to json, if not json)
            model_name (str): Name of the model.
            rhost (str): RabbitMQ host.
            rport (str): RabbitMQ port.
            user (str): RabbitMQ user.
            password (str): RabbitMQ password.
            
    Returns:
        response: Response from the backend. 
    """


    if data is None:
        data = json.dumps({})
    post_data(data, url=url, queue_name=model_name, host=rhost, port=rport, user=user, password=password)
    response = requests.post(f"{url}/{model_name}/predict")
    return response
    

def flush_rabbit(url, queue):
    response = requests.post(f"{url}/flush/{queue}")
    print(f"Response: {response.content}")


def compile_config(conf_path, conf_name):
    """Compiles the config file into a dictionary from several other .yaml configurations with hydra.

    Parameters:
            conf_path (str): Path to the configuration files.
            conf_name (str): Name of the configuration file.
    
    Returns:
            dict: Configuration dictionary

            """
    
    with initialize(config_path=conf_path):
            cfg = compose(config_name=conf_name)
    conf = OmegaConf.to_container(cfg, resolve=True)
    return conf



