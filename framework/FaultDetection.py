import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from plotly import graph_objects as go
import logging
from models.fault_detection.DBSCANAnomalyDetection import DBSCANAnomalyDetection
from models.fault_detection.EllipticEnvelopeAnomaly import EllipticEnvelopeAnomalyDetection
from models.fault_detection.IsolationForestAnomaly import IsolationForestAnomaly
from models.fault_detection.PCAAnomalyDetection import PCAAnomalyDetection
from framework.Experiment import Experiment
import yaml
from yaml import SafeLoader
from typing import Tuple


class FaultDetectionExperiment(Experiment):
    def __init__(self, cfg:dict, experiment_id:str, run_id:str, *args, **kwargs) -> None:
        """Initialize the model registry. Currently cannot be changed after initialization. In future versions, we will allow for dynamic model loading through configuration files.
        This experiment file is used for fault detection experiments. 
        It is a subclass of the Experiment class in the framework.Experiment module. It is used to create, train, and predict using fault detection models.
        Heavily relies on outlier detection/clustering algorithms.

        Parameters:
            cfg: dict The configuration file for the experiment.
            experiment_id: str The experiment id for the experiment.
            run_id: str The run id for the experiment.
        
        Implemented models:
        - DBSCAN :  'models.fault_detection.DBSCANAnomalyDetection'
        - Elliptic Envelope  'models.fault_detection.EllipticEnvelopeAnomalyDetection'
        - Isolation Forest  'models.fault_detection.IsolationForestAnomaly'
        - PCA 'models.fault_detection.PCAAnomalyDetection'

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

        ```
        The function will then be called in the system by calling the 'run' function.
        experiment.run(data, experiment_id, run_id)

        """

        self.cfg = cfg
        self.experiment_id = experiment_id
        self.run_id = run_id
        self._model_registry["dbscan"] = DBSCANAnomalyDetection
        self._model_registry["ee"] = EllipticEnvelopeAnomalyDetection
        self._model_registry["iforest"] = IsolationForestAnomaly
        self._model_registry["pca"] = PCAAnomalyDetection
        

    def setup(self, data:str) -> pd.DataFrame:
        """Setup the data for training and prediction. This function is called before any other function to set self.data that can be used in any other function. Also returns the data if need be.
        
        Parameters:
            data (str): The data to be used for training and prediction.

        Returns:
            pd.DataFrame The data that was set for the experiment.
        """
        cfg = self.cfg["setup"]
        self.ds = cfg.get("datetime_column", None)
        self.format = cfg.get("datetime_format", None)
        self.data_format =  cfg.get("format", None)
        data = self.format_data(data, self.data_format)

        self.predict_window = cfg.get("predict_window", 0)
        if self.predict_window is None:
            self.predict_window = 0
        if self.predict_window < 0:
            self.predict_window *= -1

        self.data = data
        if self.ds is not None:
            self.data = self.convert_datetime(self.data, format = self.format)
            self.data.set_index(self.ds, inplace=True) 
            
        return data
 
    def create_model(self, *args, **kwargs):
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

        self.model = model_class(**params).fit(self.data)
        self._report_registry = self.model._figs

        with mlflow.start_run(nested=True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
            mlflow.set_tag("model", model)

            for k, v in self.model._figs.items():
                v.write_html(os.path.join(os.getcwd(),"runs", self.run_id, "reports", f"{k}.html"))
                mlflow.log_artifact(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"{k}.html"), "reports")

            #mlflow.sklearn.log_model(self.model, "model")
            for k, v in params.items():
                mlflow.log_param(k, v)

        return self.model
    

    def predict(self, data:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict using the trained model. The model should be trained before calling this function.   
        Parameters:
            data (str): The data to be used for prediction.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] The predictions made by the model and the metrics calculated for the model.
        """
        data = self.format_data(data, self.data_format)
        if self.model is None:
            raise ValueError("Model not found. Please train the model first with 'create_model'.")  
        X = data.copy()

        if self.ds is not None:
            X = self.convert_datetime(X, format = self.format)
            X.set_index(self.ds, inplace=True)

        y = self.model.predict(X)
        self.new_data = pd.concat([self.new_data, y], axis=0)
        if self.predict_window > self.new_data.shape[0] :
            self.model.update_predict(self.new_data, reset_fig = True, update_fig = True)
        else:
            self.model.update_predict(self.new_data.iloc[-self.predict_window:, :], reset_fig = True, update_fig = True) 

        return y, pd.DataFrame([])

    def _score(self, y, y_hat):
        """Score the model using the predictions. This function is called after the predictions are made.
        Parameters:
            y: pd.Series The actual values.
            y_hat: pd.Series The predicted values.
        
        Returns:
            pd.DataFrame The metrics calculated for the model.
        """
        logging.log("Scoring function not implemented.")
        return pd.DataFrame([])



if __name__ == "__main__":
    import yaml
    from yaml import SafeLoader
    from omegaconf import OmegaConf
    
    with open("fault_detection_template/train/fault_detection_pca.yaml", "r") as f:
         cfg = yaml.load(f, Loader=SafeLoader)

    
    data = pd.read_excel("./data/pump_train.xlsx")
    test_data = pd.read_excel("./data/pump_test_0.xlsx")
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

        experiment = FaultDetectionExperiment(cfg=cfg)
        experiment.run(data, experiment_id, run_id)
        
        y, metrics = experiment.predict(test_data)
        experiment.save()
        
        experiment2 = FaultDetectionExperiment(cfg=cfg)
        n_exp = experiment2.load(experiment.run_id)
        
    #n_exp = TimeSeriesAnomalyExperiment(dict()).load("9b4cba6b456a4d95bd0e2c483ee12fff")
    ny, nmerics = n_exp.predict(test_data) #9b4cba6b456a4d95bd0e2c483ee12fff
    print("Done!")