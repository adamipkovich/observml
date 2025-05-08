import os
import pandas as pd
import numpy as np

import mlflow
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from framework.Experiment import Experiment
from models.fault_isolation.DecisionTree import DecisionTreeModel
from models.fault_isolation.RandomForest import RandomForestModel
from models.fault_isolation.NaiveBayes import NaiveBayesModel
from models.fault_isolation.BayesNet import BayesNet
from models.fault_isolation.HMM import HMM
from models.fault_isolation.MarkovChain import MarkovChainModel  

import logging
from copy import deepcopy
from mlflow.tracking import MlflowClient


class FaultIsolationExperiment(Experiment):
    """This experiment file is used for fault isolation experiments.
    It is a subclass of the Experiment class in the framework.Experiment module. It is used to create, train, and predict using fault isolation models.
    Heavily relies on classification algorithms.

    Implemented models:

    - Decision Tree : 'models.fault_isolation.DecisionTree'

    - Random Forest :  'models.fault_isolation.RandomForest'

    - Naive Bayes : 'models.fault_isolation.NaiveBayes'

    - HMM :  'models.fault_isolation.HMM'

    - Markov Chain :  'models.fault_isolation.MarkovChain'

    Inherits from Experiment Protocol class from framework.Experiment.

    
    If the experiment is overloaded, and new functions are added, one can call it in the system by adding the function name to the cfg file with relevant parameters.
    For example, if a new function 'new_function' is added to the system, the cfg file should have the following structure:
    cfg file should have the following structure:
    

    ```
    load_object:
        module: framework.FaultIsolation
        name: FaultIsolationExperiment
    setup:
        datetime_column: str The column that contains the datetime information.
        target: str The target column for the experiment. (output column)
    ...
    eda:
    create_model:
        model: str The model to be used for training.
        params: dict The parameters to be used for the model.
    new_function:
        param1: str The first parameter for the function.
        param2: str The second parameter for the function.
    ...
    ```
    """
    
    def __init__(self, cfg:dict, experiment_id:str, run_id:str, * args, **kwargs) -> None:
        """Initialize the model registry. Currently cannot be changed after initialization. In future versions, we will allow for dynamic model loading through configuration files.
        
        Parameters:

            cfg (dict): The configuration file for the experiment.
            experiment_id (str): The experiment id for the experiment.
            run_id (str): The run id for the experiment.
        """
        
        self.cfg = cfg
        self.experiment_id = experiment_id
        self.run_id = run_id
        self._model_registry["dt"] = DecisionTreeModel # decision tree
        self._model_registry["rf"] = RandomForestModel # random forest
        self._model_registry["nb"] = NaiveBayesModel # Naive Bayes
        self._model_registry["bn"] = BayesNet # Bayes Net!
        self._model_registry["hmm"] = HMM # hidden markov model
        self._model_registry["mc"] = MarkovChainModel# Markov Chain
        
    def setup(self, data:str) -> pd.DataFrame:
        """Setup the data for training and prediction. This function is called before training the model. 
        Parameters:

            data (str): The data to be used for training and prediction in json format.
        
        Returns:
            data (pd.DataFrame): The data set up for training and prediction.
        """
        cfg = self.cfg["setup"]

        self.data_format =  cfg.get("format", None)
        data = self.format_data(data, self.data_format)

        self.data = data
        print(self.data)
       
        self.ds = cfg.get("datetime_column", None)
        self.format = cfg.get("datetime_format", None)
        self.predict_window = cfg.get("predict_window", 0)
        if self.predict_window is None:
            self.predict_window = 0
        if self.predict_window < 0:
            self.predict_window *= -1
        retrain_cfg = cfg.get("retrain", None)

        if retrain_cfg is None or len(retrain_cfg) == 0:
            self.retrain_window = 0
            self.metric = None
            self.metric_threshold = 0.0
            self.higher_better = True
        else:
            self.retrain_window = retrain_cfg.get("retrain_window", 0)
            self.metric = retrain_cfg.get("metric", None)
            self.metric_threshold = retrain_cfg.get("metric_threshold", 0.0)
            self.higher_better = retrain_cfg.get("higher_better", True)

       
        self.target = cfg["target"]
        # self.target = self.target.replace(" ", "_")   # replace spaces with underscores
        # self.target = self.target.replace("//", "") 
        # self.target = self.target.replace("/", "") 
        # self.target = self.target.replace("(", "")
        # self.target = self.target.replace(")", "")
        # self.target = self.target.replace("\\", "")
        # self.target = self.target.replace(".", "")

        
        ## TODO: Abstrct data VC -- integrate dvc or other data versioning tools.
        if self.ds is not None:
            self.data = self.convert_datetime(self.data, format = self.format)
            self.data.set_index(self.ds, inplace=True)

        self.data.rename(columns={self.target : "target"}, inplace=True)

        self.data["target"] = self.data["target"].astype("category")
        self.input_scheme = self.data.columns.to_list()
        self.input_scheme.remove("target")
        
        return data

    def create_model(self, *args, **kwargs) -> any: 
        """Create the model using the configuration file. The model is trained on the data set up in the 'setup' function.

        Returns: 
            (any) The trained model."""
    
        
        model = self.cfg["create_model"]["model"]
        params = self.cfg["create_model"].get("params", None)
        if params is None:
            params = dict()

        if self.data is None:
            raise ValueError("Data not found. Please use 'setup' to setup the data first.")

        try:
            model_class = self._model_registry[model]
        except KeyError:
            raise ValueError(f"Model {model} not found in model registry. Please check configuration file. Available models are {self._model_registry.keys()}.")
        self.model = model_class(**params).fit(self.data)
        self._report_registry = self.model._figs
        y = self.model.predict(self.data)
        try:
            self.metrics = self._score(self.data["target"], y["y_pred"])
            self._report_registry["metrics"] = px.line(self.metrics, x = self.metrics.index, y = self.metrics.columns, markers=True)
        except Exception as e:
            self.metrics = pd.DataFrame([np.NaN, np.NaN, np.NaN, np.NaN], index = ["Accuracy", "F1", "Precision", "Recall"]).transpose()
            logging.warn(f"Error calculating metrics: {e}. Model {self.model.__class__.__name__} may not have y_pred as output.")

        with mlflow.start_run(nested=True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
            mlflow.set_tag("model", model)
            for k, v in self.model._figs.items():
                v.write_html(os.path.join(os.getcwd(),"runs", self.run_id, "reports", f"{k}.html"))
                mlflow.log_artifact(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"{k}.html"), "reports")
            for k, v in params.items():
                mlflow.log_param(k, v)

        return self.model
                

    def predict(self, data:str):
        """Predict using the trained model. The model should be trained before calling this function.
        
        Parameters:
            data (str): The data to be used for prediction in json format.

        Returns:
            (pd.DataFrame) The predictions made by the model."""
        data = self.format_data(data, self.data_format)
        if self.model is None:
            raise ValueError("Model not found. Please train the model first with 'create_model'.")  
        X = data.copy()
        if self.ds is not None:
            X = self.convert_datetime(X, format = self.format)
            X.set_index(self.ds, inplace=True)

        if self.target in X.columns:
            X.rename(columns={self.target : "target"}, inplace=True)
        y = self.model.predict(X)

        metrics = self._score(X["target"], y["y_pred"])
        self.metrics = pd.concat([self.metrics, metrics], axis=0, ignore_index=True)
        self.new_data = pd.concat([self.new_data, y], axis=0)

        if self.predict_window > self.new_data.shape[0] :
            self.model.update_predict(self.new_data, reset_fig = True, update_fig = True)
        else:
            self.model.update_predict(self.new_data.iloc[-self.predict_window:, :], reset_fig = True, update_fig = True) 

        self._report_registry["metrics"] = px.line(self.metrics, x = self.metrics.index, y = self.metrics.columns, markers=True)
        return y, metrics

    def _score(self, y, y_hat):
        """Score the model using the predictions. This function is called after the predictions are made.
        Parameters:
            y (pd.Series): The actual values.
            y_hat (pd.Series): The predicted values.
        
        Returns:
            (pd.DataFrame) The metrics calculated for the model."""


        if self.model is None:
            raise ValueError("Model not found. Please train the model first with 'create_model'.")
        if y is None:
            raise ValueError("No data found for scoring. Please provide data for scoring.")
        if y_hat is None:
            raise ValueError("No predictions found for scoring. Please provide predictions for scoring.")
        
        acc = accuracy_score(y, y_hat)
        f1 = f1_score(y, y_hat)
        prec = precision_score(y, y_hat)
        rec = recall_score(y, y_hat)
        return pd.DataFrame([acc, f1, prec, rec], index=["Accuracy", "F1", "Precision", "Recall"]).transpose()

        

## Testing 
if __name__ == "__main__":
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    
    conf_path = "./configs"
    conf_name = "config"
    with initialize(config_path=conf_path):
            cfg = compose(config_name=conf_name)
    conf = OmegaConf.to_container(cfg, resolve=True)

    cfg = conf["detect"]
    data = pd.read_excel("./data/detect_train.xlsx")
    
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "proba"
    client = MlflowClient("http://localhost:5000")
    try:
         client.create_experiment(experiment_name)
    except Exception as e:
         pass    
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    with mlflow.start_run(experiment_id = experiment_id) as run:

        experiment_id = run.info.experiment_id
        run_id = run.info.run_id

        if not os.path.exists(os.path.join(os.getcwd(), "runs")):
            os.makedirs(os.path.join(os.getcwd(), "runs"))
        if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id)):
            os.makedirs(os.path.join(os.getcwd(), "runs", run_id))
        if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id, "reports")):
            os.makedirs(os.path.join(os.getcwd(), "runs", run_id, "reports"))
        if not os.path.exists(os.path.join(os.getcwd(), "runs",run_id, "model")):
            os.makedirs(os.path.join(os.getcwd(), "runs", run_id, "model"))


        experiment = FaultIsolationExperiment(cfg=cfg, experiment_id=  experiment_id, run_id=run_id)
        experiment.run(data=data)
        test_data = pd.read_excel("./data/detect_test_0.xlsx")
        y, metrics = experiment.predict(test_data)
        test_data = pd.read_excel("./data/detect_test_1.xlsx")
        y, metrics = experiment.predict(test_data)
        experiment._report_registry["Vc_prediction"].show()
        experiment.save()
        
        experiment2 = FaultIsolationExperiment(cfg=cfg)
        n_exp = experiment2.load(experiment.run_id)
        
    #n_exp = TimeSeriesAnomalyExperiment(dict()).load("9b4cba6b456a4d95bd0e2c483ee12fff")
    ny, nmerics = n_exp.predict(test_data) #9b4cba6b456a4d95bd0e2c483ee12fff
    print("Done!")

