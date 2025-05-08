import os
import pandas as pd
import numpy as np
import mlflow
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from framework.Experiment import Experiment
from models.spmf.Apriori import Apriori
from models.spmf.CM_SPAM import CMSPAM
from models.spmf.TopKRules import TopKRules
from models.spmf.HeuristicsMiner import HeuristicsMiner

from mlflow.tracking import MlflowClient
import yaml
from yaml import SafeLoader

class ProcessMiningExperiment(Experiment):
    """ This experiment file is used for process mining experiments.
    It is a subclass of the Experiment class in the framework.Experiment module. It is used to create, train, and predict using process mining models.
    Heavily relies on sequence mining algorithms.
    
    Implemented models:
    - Apriori : 'models.spmf.Apriori'
    - CMSPAM :  'models.spmf.CM_SPAM'
    - TopKRules :'models.spmf.TopKRules'
    - Heuristics Miner : 'models.spmf.HeuristicsMiner'
    Inherits from Experiment Protocol class from framework.Experiment.
    If the experiment is overloaded, and new functions are added, one can call it in the system by adding the function name to the cfg file with relevant parameters.
    For example, if a new function 'new_function' is added to the system, the cfg file should have the following structure:
    cfg file should have the following structure:
    ```
    load_object:
        module: framework.ProcessMining    
        name: ProcessMiningExperiment
    setup:
    ...
    create_model:
        model: str The model to be used for training.
        params: dict The parameters to be used for the model.
    new_function:
        param1: str The first parameter for the function.
        param2: str The second parameter for the function.
    ...
    ```


    """


    def __init__(self, cfg:dict,  experiment_id:str, run_id:str, * args, **kwargs) -> None:
        """Initialize the model registry. Currently cannot be changed after initialization.
        Parameters:
            cfg: dict The configuration file for the experiment.
            experiment_id: str The experiment id for the experiment.
            run_id: str The run id for the experiment.
        """
        
        self.cfg = cfg
        self.experiment_id = experiment_id
        self.run_id = run_id
        self._model_registry["apriori"] = Apriori
        self._model_registry["cmspam"] = CMSPAM
        self._model_registry["topk"] = TopKRules
        self._model_registry["heuristics"] = HeuristicsMiner

        
    def setup(self, data:pd.DataFrame):
        """Setup the data for training and prediction. This function is called before training the model. Data must be in pandas DataFrame format - can be a columns, with several rows. 
        
        Parameters:
            data: pd.DataFrame The data to be used for training and prediction.

        Returns:
            pd.DataFrame The data after processing. This data is used for training and prediction    """
        
        cfg = self.cfg["setup"]
        self.data_format =  cfg.get("format", None)
        data = self.format_data(data, self.data_format)
        # col_names = dict("Start_timestamp" : "start:timestamp",
        #                 "End_timestamp" : "time:timestamp",
        #                 "Event" : "concept:name",
        #                 "Case_id" : "case:concept:name",
        #                 "Resource" : "org:resource",
        #                 "Ordered" : "Ordered",
        #                 "Completed" : "Completed",
        #                 "Rejected" : "Rejected",
        #                 "MRB" : "MRB",
        #                 "Part" : "Part")
        #data.rename(columns=col_names, inplace=True)
        
        #TODO : excavate a log ...
       
        self.data = data#.drop(columns=[self.ds])
        return data

    def create_model(self, *args, **kwargs)->any:
        """Create the model using the configuration file. The model is trained on the data set up in the 'setup' function.
        Returns:
            any The trained model.
        """
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

        ## data.to_csv()

        self.model = model_class(**params).fit(self.data)
        self._report_registry = self.model._figs

        with mlflow.start_run(nested=True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
            mlflow.set_tag("model", model)
            for k, v in self.model._figs.items():
                v.write_html(os.path.join(os.getcwd(),"runs", self.run_id, "reports", f"{k}.html"))
                mlflow.log_artifact(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"{k}.html"), "reports")

            
            for k, v in params.items():
                mlflow.log_param(k, v)

        return self.model
                

    def predict(self, data:pd.DataFrame):
        """Predict using the trained model. The model should be trained before calling this function.
        Parameters:
            data: pd.DataFrame The data to be used for prediction.

        Returns:
            pd.DataFrame The predictions made by the model.
        """
        if self.model is None:
            raise ValueError("Model not found. Please train the model first with 'create_model'.")  
        X = data.copy()
        y = self.model.predict(X)
        return y, pd.DataFrame([])

    def _score(self, y, y_hat):
        """Score the model using the predictions. This function is called after the predictions are made.
        Parameters:
            y: pd.Series The actual values.
            y_hat: pd.Series The predicted values.
        Returns:
            pd.DataFrame The scores for the model."""

        if self.model is None:
            raise ValueError("Model not found. Please train the model first with 'create_model'.")
        if y is None:
            raise ValueError("No data found for scoring. Please provide data for scoring.")
        if y_hat is None:
            raise ValueError("No predictions found for scoring. Please provide predictions for scoring.")
        
        return pd.DataFrame([])

        
if __name__ == "__main__":
    import yaml
    from yaml import SafeLoader
    from omegaconf import OmegaConf
    
    with open("text_mining_template/train/ktop.yaml", "r") as f:
         cfg = yaml.load(f, Loader=SafeLoader)

    data = pd.read_csv("./data/traces_csoft_oper.csv")
    exp = ProcessMiningExperiment(cfg)
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

        experiment = TextMiningExperiment(cfg=cfg)
        experiment.run(data, experiment_id, run_id)
        
        y, metrics = experiment.predict(data)
        experiment.save()
        
        experiment2 = TextMiningExperiment(cfg=cfg)
        n_exp = experiment2.load(experiment.run_id)
        
    #n_exp = TimeSeriesAnomalyExperiment(dict()).load("9b4cba6b456a4d95bd0e2c483ee12fff")
#9b4cba6b456a4d95bd0e2c483ee12fff
    print("Done!")