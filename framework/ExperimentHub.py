import pika.exceptions
from framework.Experiment import Experiment, load_object
import mlflow
from mlflow.tracking import MlflowClient
import pika
import yaml
from yaml import SafeLoader
import pandas as pd
import os
import logging
from asyncio import sleep
import time
from copy import deepcopy
class ExperimentHub:
    """Class to manage multiple experiments. Abstracts from the user the need to manage multiple experiments. Must have a RabbitMQ and MLflow server running. Otherwise it fails.
    :attr experiments: Dictionary to store experiments.
    :attr run_ids: Dictionary to store run IDs.
    :attr experiment_ids: Dictionary to store experiment IDs.
    :attr available: Dictionary to store availability of experiments.
    :attr rabbit_connection: RabbitMQ connection (pika.BlockingConnection).
    :attr channel: RabbitMQ channel. 

    ::todo::
        - add Postgres support for data storage - new data, metrics, params, logs etc.
        - celery for better concurrency managent
        - add email sending if and event system to notify user of retraining/anomalies etc.
        - add logging
        - add retraining functionality
        - git integration for version control + pulling new models from git so that the image does not need to be rebuilt.
    """

    experiments : dict[str, Experiment] = dict() ## rabbit queue is name of experiment
    run_ids : dict[str, str] = dict()
    experiment_ids : dict[str, str] = dict()
    available : dict[str, bool] = dict()
    rabbit_connection = None ## blocking connection 
    channel = None ## blocking channel

    def __init__(self, mlflow_uri:str, rabbit_host:str, rabbit_port:str|int, rabbit_user:str, rabbit_password:str) -> None:
        """Constructor for ExperimentHub class. Initializes RabbitMQ connection and MLflow tracking URI. 
        :param mlflow_uri: URI for MLflow tracking server. --> will most possbibly be http://localhost:5000, or defined by docker networking. 
        :param rabbit_host: Hostname for RabbitMQ server. --> will most possbibly be localhost, or defined by docker networking.
        :param rabbit_port: Port for RabbitMQ server. --> will most possbibly be 5672, or defined by docker port mapping.
        :param rabbit_user: Username for RabbitMQ server.
        :param rabbit_password: Password for RabbitMQ server.
        :return: None"""
        self.mlflow_uri = mlflow_uri
        self.rabbit_host = rabbit_host
        self.rabbit_port = rabbit_port
        self._mlflow_tracking_uri()
        self._connect_to_rabbit(rabbit_host, rabbit_port, rabbit_user, rabbit_password)
        
    
    def load(self, name:str, run_id:str = None) -> None:
        """Function to load saved experiment. 
        :param name: Name of the experiment.
        :param run_id: Run ID of the experiment.
        :return: None
        """

        if name in self.run_ids.keys():
            run_id = self.run_ids[name]
        else:
            if run_id is None:
                logging.error("Run ID is not specified.")
                return "Run id was not found."
            else:
                self.run_ids[name] = run_id
        

        if not os.path.exists(os.path.join(os.getcwd(), run_id)):
            os.makedirs(os.path.join(os.getcwd(), run_id))
            mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/metadata.yaml", dst_path=os.path.join(os.getcwd(), "runs", run_id) )
        
        with open(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml")) as f: 
            cfg = yaml.load(f, Loader=SafeLoader) 

        interface = cfg["load_object"]["name"]
        module = cfg["load_object"]["module"]

        interface_class = load_object(module, interface)
        run = mlflow.get_run(run_id)
        exp_id = run.info.experiment_id
        experiment = interface_class(cfg = cfg, run_id = run_id, experiment_id =exp_id )
        experiment = experiment.load(run_id=run_id)
        self.experiments[name] = experiment
        self.available[name] = True

    def save(self, name:str) -> None:
        """Function to save experiment. Calls experiment.save() method.
        :param name: Name of the experiment.
        :return: None"""
        #TODO: save experiment to disk
        try:
            self.experiments[name].save()   
        except Exception as e:
            logging.error(e)

    
    def kill(self, name:str, **kwargs): 
        """Function to kill experiment. Deletes experiment from registry without saving."""
        ## TODO: add save/update features etc...
        try:
            self.experiments.pop(name) 
        except Exception as e:
            logging.error(e)
    
    async def train(self, name:str, cfg:dict):
        """Function to run experiment. This is an async function that runs the experiment sequence defined by the config file in a separate thread. 
        
        :param name: Name of the experiment (key in self.experimets).
        :param cfg: Configuration dictionary for the experiment.
        
        :return: None

        ::todo::
            - add celery for concurrency
            - add logging for errors/states
            - add eda test - do not go through the training process if the data is not clean/or causes errors.
            
        """

        self.available[name] = False
        
        interface = cfg["load_object"]["name"]
        module = cfg["load_object"]["module"]
        experiment_class = load_object(module, interface)
        client = MlflowClient(self.mlflow_uri)
        try:
            client.create_experiment(name)
        except Exception as e:
            pass    
        experiment_id = client.get_experiment_by_name(name).experiment_id
        self._create_queue(name)

        # Feature names unseen at fit time:
        # - Output_S --
        # - ds
        # Feature names seen at fit time, yet now missing:
        # - y_pred


        data = self._pull_from_rabbit(name)
        
        with mlflow.start_run(experiment_id=experiment_id):
            #experiment_id = mlflow.active_run().info.experiment_id
            run_id = mlflow.active_run().info.run_id
            experiment = experiment_class(cfg = cfg, run_id = run_id, experiment_id = experiment_id)
            if not os.path.exists(os.path.join(os.getcwd(), "runs")):
                os.makedirs(os.path.join(os.getcwd(), "runs"))
            if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id)):
                os.makedirs(os.path.join(os.getcwd(), "runs", run_id))
            if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id, "reports")):
                os.makedirs(os.path.join(os.getcwd(), "runs", run_id, "reports"))
            if not os.path.exists(os.path.join(os.getcwd(), "runs",run_id, "model")):
                os.makedirs(os.path.join(os.getcwd(), "runs", run_id, "model"))
            
            #yaml.dump(metadata, open(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml"), "w"))  
            #mlflow.log_artifact(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml"), "")
            experiment.run(data)
            experiment.save()

        self.experiments[name] = experiment
        self.run_ids[name] = run_id
        self.experiment_ids[name] = experiment_id
        self.available[name] = True
        return True
    
    async def retrain(self, name : str):
        if self.experiments.get(name, None) is None:
            return
        
        self.available[name] = False
        cfg = deepcopy(self.experiments[name].cfg)
        metrics = self.experiments[name].metrics.copy()
        inputs = self.experiments[name].input_scheme
        experiment_class = self.experiments[name].__class__
        data = self.experiments[name].join_data()

        if hasattr(self.experiments[name], "ds"):
            if "ds" not in data.columns:
                data.reset_index(drop=False, names = self.experiments[name].ds, inplace= True)
            inputs.append(self.experiments[name].ds)

        if hasattr(self.experiments[name], "target"):
            data.rename(columns={"target" : self.experiments[name].target}, inplace=True)
            inputs.append(self.experiments[name].target)


        data = data.loc[:, inputs]
        client = MlflowClient(self.mlflow_uri)  
        experiment_id = client.get_experiment_by_name(name).experiment_id
        self._create_queue(name)

        with mlflow.start_run(experiment_id=experiment_id):
            #experiment_id = mlflow.active_run().info.experiment_id
            run_id = mlflow.active_run().info.run_id
            experiment = experiment_class(cfg = cfg, run_id = run_id, experiment_id = experiment_id)
            if not os.path.exists(os.path.join(os.getcwd(), "runs")):
                os.makedirs(os.path.join(os.getcwd(), "runs"))
            if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id)):
                os.makedirs(os.path.join(os.getcwd(), "runs", run_id))
            if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id, "reports")):
                os.makedirs(os.path.join(os.getcwd(), "runs", run_id, "reports"))
            if not os.path.exists(os.path.join(os.getcwd(), "runs",run_id, "model")):
                os.makedirs(os.path.join(os.getcwd(), "runs", run_id, "model"))
            
            #yaml.dump(metadata, open(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml"), "w"))  
            #mlflow.log_artifact(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml"), "")
            experiment.run(data)
            experiment.save()

        experiment.metrics = metrics
        self.experiments[name] = experiment
        self.run_ids[name] = run_id
        self.experiment_ids[name] = experiment_id
        self.available[name] = True
        return True


    def plot(self, name:str, plot_name:str):
        """Function to get figures. From experiment {name} get figure {plot_name}. Used by API.
        :param name: Name of the experiment.
        :param plot_name: Name of the plot.
        
        :return: go.Figure object
        """
        if name not in self.experiments.keys() or not self.available[name] or name == 'None' or plot_name == 'None':
            return "Experiment not found." ## TODO: add empty registry with go.Figure() object
        
        return self.experiments[name]._report_registry[plot_name] 
       
    
    def plot_names(self, name:str) -> list[str] | str:
        """Function to get all figure names. Used by API.
        
        :param name: Name of the experiment.
        :return: List of figure names.
        """
        if self.experiments and name != 'None' and self.experiments[name]:
            return [list(self.experiments[name]._report_registry.keys()), list(self.experiments[name]._eda_registry.keys())]
        else:
            return "No reports available."

    async def predict(self, name:str)-> pd.DataFrame:  
        """Function to predict experiment. It is an async function that pulls data from RabbitMQ and calls experiment.predict() method.
        :param name: Name of the experiment.
        :return: pd.DataFrame Prediction from the experiment."""
        #experiment = self.experiments[name]
        #X = self._pull_from_rabbit(name)
        #return experiment.predict(X)
        #try:
        available = self.available[name]
        while not available:
            await sleep(0.2)
            available = self.available[name]
            logging.info(f"Waiting for {name} to be available.")

        experiment = self.experiments[name]
        X = self._pull_from_rabbit(name)
        y = experiment.predict(X)

        if experiment.retrain():
                await self.retrain(name)
        return y
        
        # except Exception as e:
        #      logging.error(e)
        #      return None
        
    def get_train_data(self, name:str) -> str:
        """Function to get training data.
        :param name: Name of the experiment.
        :return: str Training data from the experiment in json format."""

        if name not in self.experiments.keys() or not self.available[name]:
            return "Experiment not found."
        return self.experiments[name].data.reset_index(inplace=False).to_json()

    def _create_queue(self, queue_name:str) -> None:
        """Function to create a queue in RabbitMQ. 
        :param queue_name: Name of the queue.
        :return: None
        
        ::todo::
            - add error handling
            - log queue
        """

        self.channel.queue_declare(queue=queue_name, durable = True)

    def get_cfg(self, name:str):
        """Function to get configuration of the experiment. 
        :param name: Name of the experiment.
        :return: dict Configuration of the experiment."""

        if name not in self.experiments.keys() or not self.available[name]:
           Warning(f"Experiment {name} not found.")
           return {} 
        return self.experiments[name].cfg
    

    def _connect_to_rabbit(self, host:str, port : str|int, username:str, password:str) -> None:
        """Function to connect to RabbitMQ server.
        :param host: Hostname of the RabbitMQ server.
        :type host: str
        :param port: Port of the RabbitMQ server.
        :type port: str|int
        :param username: Username for the RabbitMQ server.
        :type username: str
        :param password: Password for the RabbitMQ server.
        :type password: str
        :return: None
        """

        #self.rabbit_username = username ## TODO: SHA256
        #self.rabbit_password = password
        if self.rabbit_connection is not None:
            if self.rabbit_connection.is_closed:
                self.rabbit_connection = None

        if self.rabbit_connection is None:
            credentials = pika.PlainCredentials(username=username, password=password)
            while self.rabbit_connection is None:
                try:
                    self.rabbit_connection = pika.BlockingConnection(pika.ConnectionParameters(host = host, port = port, credentials=credentials, heartbeat=0))
                except pika.exceptions.AMQPConnectionError:
                    logging.error(f"Connection to RabbitMQ failed at {host}:{port}. Retrying...")
                    time.sleep(3)
                
                    ## retry 5 second

                ## retry 5 second
            ## TODO: TIDY UP THE RABBIT CODE
            self.channel = self.rabbit_connection.channel()
            self.channel.basic_qos(prefetch_count=1)

    
    def _disconnect_from_rabbit(self) -> None:
        """Function to disconnect from RabbitMQ server.
        :return: None
        """
        if self.channel is not None:
            if self.channel.is_open:
                self.channel.close()
                self.channel = None

        if self.rabbit_connection is not None:
            if self.rabbit_connection.is_open:
                self.rabbit_connection.close()
                self.rabbit_connection = None
        return
    
    def _pull_from_rabbit(self, queue:str) -> pd.DataFrame:
        """Function to pull data from RabbitMQ server.
        :param queue: Name of the queue.
        :return: pd.DataFrame Data from the queue."""
       
        method_frame, header_frame, body = self.channel.basic_get(queue)
        if method_frame:
            data = body.decode("utf-8") #pd.read_json(body.decode("utf-8"))
            self.channel.basic_ack(method_frame.delivery_tag)
            #cols = data.columns
            # cc = [i.replace(" ", "_") for i in cols] # NOTE: cols must be renamed in Experiments as well
            # cc = [i.replace("//", "") for i in cc]
            # cc = [i.replace("/", "") for i in cc]
            # cc = [i.replace("\\", "") for i in cc] 
            # cc = [i.replace("(", "") for i in cc] 
            # cc = [i.replace(")", "") for i in cc] 
            # cc = [i.replace(".", "") for i in cc] 
            #data.set_axis(cc, axis=1, inplace=True)
            return data        
        else:
            return "" #pd.DataFrame([])
        

    def _mlflow_tracking_uri(self) -> None:
        """Function to set MLflow tracking URI. 
        :return: None
        """
        mlflow.set_tracking_uri(self.mlflow_uri)
        
    
    def _flush(self, queue:str) -> None:
        """Function to flush queue in RabbitMQ server. This function is used when requests are not handled properly, and the queues start to pile up.
        :param queue: Name of the queue.
        :return: None
        
        ::todo::
            - mode to flush by predicting on each message
            - mode to flush by train + predicts on each message"""
        self.channel.queue_declare(queue=queue, durable=True)
        self.channel.queue_purge(queue)

    def plot_eda(self, name:str, fig_name:str):
        """Function to get EDA figure.
        :param name: Name of the experiment.
        :param fig_name: Name of the figure.
        :return: go.Figure EDA figure."""
        if name not in self.experiments.keys() or not self.available[name] or name == 'None' or fig_name == 'None':
            return "Experiment not found."
        
        return self.experiments[name]._eda_registry[fig_name]