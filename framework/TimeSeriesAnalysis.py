import os
import sys
import pandas as pd
import numpy as np
import mlflow
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from models.time_series_analysis.Autoencoder import Autoencoder, AnomalyDetector
from models.time_series_analysis.LSTM import LSTM
from models.time_series_analysis.Prophet import ProphetAnomalyDetection
from models.time_series_analysis.SSA import SSAAnomalyDetection
from models.time_series_analysis.ARIMA import ARIMAAnomalyDetector
from models.time_series_analysis.ExponentialSmoothing import ExponentialSmoothingAnomaly
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from framework.Experiment import Experiment
import logging
from mlflow.tracking import MlflowClient



class TimeSeriesAnomalyExperiment(Experiment):
    """Class for interacting with time series models. Plays a similar role to pycaret experiment. Deals with model training, predicting, metrics and serialization.
    Inherits from Experiment Protocol class from framework.Experiment.
    Implemented models:
    - Autoencoder : 'models.time_series_analysis.Autoencoder'
    - LSTM :  'models.time_series_analysis.LSTM'
    - Prophet :  'models.time_series_analysis.Prophet'
    - SSA : 'models.time_series_analysis.SSA'
    - ARIMA : 'models.time_series_analysis.ARIMA'
    - Exponential Smoothing : 'models.time_series_analysis.ExponentialSmoothing'


    If the experiment is overloaded, and new functions are added, one can call it in the system by adding the function name to the cfg file with relevant parameters.
    For example, if a new function 'new_function' is added to the system, the cfg file should have the following structure:
    cfg file should have the following structure:

    ```
    load_object:
        module: framework.TimeSeriesAnalysis
        name: TimeSeriesAnomalyExperiment
    setup:
        ds : str The column that contains the datetime information.
        y : str The target column for the experiment. (output column)
    ...
    create_model:
        model: str The model to be used for training.
        params: dict The parameters to be used for the model.
    ...
    new_function: # e.g. posterior reconciliation of results.
        param1: str The first parameter for the function.
        param2: str The second parameter for the function.
    ```
    """

    def __init__(self, cfg:dict, experiment_id:str, run_id:str) -> None:
        """Initialize the model registry. Currently cannot be changed after initialization. In future versions, we will allow for dynamic model loading through configuration files.
        This experiment function is used for loading basic time series anomaly detection models.
        Parameters:
            cfg: dict The configuration file for the experiment.
            experiment_id: str The experiment id for the experiment.
            run_id: str The run id for the experiment.
        """ 
       
        self.cfg = cfg
        self.experiment_id = experiment_id
        self.run_id = run_id
        
        self._model_registry["ae"] = Autoencoder
        self._model_registry["lstm"] = LSTM
        self._model_registry["prophet"] = ProphetAnomalyDetection
        self._model_registry["ssa"] = SSAAnomalyDetection
        self._model_registry["arima"] = ARIMAAnomalyDetector
        self._model_registry["es"] = ExponentialSmoothingAnomaly


    def setup(self, data:str, *args, **kwargs):
        """Setup the data for training and prediction. This function is called before training the model. In the future, it will also be used to preprocess the data and prepare it for training.
        Parameters:
            data: pd.DataFrame The data to be used for training and prediction.
        
        Returns:
            pd.DataFrame The data after processing. This data is used for training and prediction.


        Description:

        - The setup function is used to prepare the data for training and prediction. It is called before the model is trained.
        - The function renames the columns to 'ds' and 'y' for consistency.
        - The function logs the target and datestamp columns to mlflow.
        - The function returns the data after processing.
        """
        cfg = self.cfg["setup"]

        self.data_format =  cfg.get("format", None)
        data = self.format_data(data, self.data_format)

        self.ds = cfg["datetime_column"]
        self.target = cfg["target"]
        self.format = cfg["datetime_format"]
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


        if self.ds not in data.columns:
            raise ValueError("Datestamp column not found in data. Configuration file must specify datestamp column as ds.")
        if self.target not in data.columns:
            raise ValueError("Target column not found in data. Configuration file must specify target column as target.")
        
        
        data = self.convert_datetime(data, format= self.format)
        data.rename(columns = {self.ds : "ds", self.target : "y"}, inplace=True)
        self.data = data.loc[:, ["ds", "y"]]
        self.input_scheme = []
        with mlflow.start_run(nested= True, experiment_id=self.experiment_id) as run:
            mlflow.log_param("target", self.target)
            mlflow.log_param("datetime_column", self.ds)

        
        return data

    def create_model(self, *args, **kwargs):
        """Create the model using the configuration file. The model is trained on the data set up in the 'setup' function.
        Returns:
            any The trained model.

        Description:    
        - The function creates the model using the configuration file.
        - The function logs the model and parameters to mlflow.
        - The function logs the metrics to mlflow.
        - The function saves the model to disk.
        - The function returns the trained model.
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
        y = self.model.predict(self.data)

        try:
            self.metrics = self._score(self.data[self.target], y["y_pred"])
            self._report_registry["metrics"] = px.line(self.metrics, x = self.metrics.index, y = self.metrics.columns, markers=True)
            
        except Exception as e:
            self.metrics = pd.DataFrame([np.NaN, np.NaN, np.NaN, np.NaN], index = ["MSE", "MAE", "R2", "MAPE"]).transpose()
            logging.warn(f"Error calculating metrics: {e}. Model {self.model} may not have y_pred as output.")

        with mlflow.start_run(nested=True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
            mlflow.set_tag("model", model)
            #mlflow.sklearn.save_model(self.model, f"runs/{self.run_id}/model")

            for k, v in params.items():
                mlflow.log_param(k, v)
            for key, value in self.metrics.to_dict().items():
                mlflow.log_metric(key, value[0])

            for k,v in self._report_registry.items():
                v.write_html(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"{k}_report.html"))
                mlflow.log_artifact(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"{k}_report.html"), "reports")

        return self.model
                

    def predict(self, data:str):
        """Predict using the trained model. The model should be trained before calling this function.
        Parameters:
            data str The data to be used for prediction.
        
        Returns:
            pd.DataFrame The predictions made by the model.
        
        Description:
        - The function predicts using the trained model.
        - The function updates the predict figures with the new data.
        - The function calculates the metrics for the model.
        - The function logs the metrics to mlflow.
        - The function saves the reports to disk.
        - The function logs the reports to mlflow.
        - The function returns the predictions and metrics.
        """
        data = self.format_data(data, self.data_format)
        if self.model is None:
            raise ValueError("Model not found. Please train the model first with 'create_model'.")  
        
        data.rename(columns = {self.ds : "ds", self.target : "y"}, inplace=True)
        data = self.convert_datetime(data, format = self.format)
        data = data.loc[:, ["ds", "y"]]
        y = self.model.predict(data)
        self.new_data = pd.concat([self.new_data, y], axis=0)

        if self.predict_window > self.new_data.shape[0] :
            self.model.update_predict(self.new_data, reset_fig = True, update_fig = True)
        else:
            self.model.update_predict(self.new_data.iloc[-self.predict_window:, :], reset_fig = True, update_fig = True) 
        if "y_pred" not in y.columns:
            metrics = pd.DataFrame([np.NaN, np.NaN, np.NaN, np.NaN], index = ["MSE", "MAE", "R2", "MAPE"]).transpose()
            logging.warn("y_pred not found in model output. Please make sure the model has a 'predict' method that returns a DataFrame with 'y_pred' column.")
        else:
            metrics = self._score(data[self.target], y["y_pred"])

        
        self.metrics = pd.concat([self.metrics, metrics], axis=0, ignore_index=True)
        self._report_registry["metrics"] = px.line(self.metrics, x = self.metrics.index, y = self.metrics.columns, markers=True)
        
        #self.spc_chart(update_fig=True)
        self._report_registry["predict"].write_html(os.path.join(os.getcwd(), "runs", self.run_id, "reports","predict_report.html"))
        self._report_registry["metrics"].write_html(os.path.join(os.getcwd(), "runs", self.run_id, "reports", "metrics_report.html"))
        with mlflow.start_run(nested=True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
            mlflow.log_artifact(os.path.join(os.getcwd(), "runs", self.run_id, "reports", "predict_report.html"), "reports")
            mlflow.log_artifact(os.path.join(os.getcwd(), "runs", self.run_id, "reports", "metrics_report.html"), "reports")

        return y, metrics
       

    def _score(self, y, y_hat):
        """Calculate the metrics for the model.
        Parameters:
            y: pd.Series The true values.
            y_hat: pd.Series The predicted values.

        Returns:
            pd.DataFrame The metrics for the model.
       
        Description:

        - The function calculates the metrics for the model.
        - The function returns:
            - Mean Squared Error
            - Mean Absolute Error
            - R2 Score
            - Mean Absolute Percentage Error
        """

        
        assert isinstance(y, pd.Series) or isinstance(y, pd.DataFrame), "y must be a pandas Series or DataFrame."
        assert isinstance(y_hat, pd.Series) or isinstance(y_hat, pd.DataFrame), "y_hat must be a pandas Series or DataFrame."
        
        if isinstance(y, pd.Series):
            y = y.to_frame()
        if isinstance(y_hat, pd.Series):
            y_hat = y_hat.to_frame()

        assert y.shape == y_hat.shape, "y and y_hat must have the same shape."   

        mse = mean_squared_error(y, y_hat)
        mae = mean_absolute_error(y, y_hat)
        r2 = r2_score(y, y_hat)
        mape = mean_absolute_percentage_error(y, y_hat)

        return pd.DataFrame([mse, mae, r2, mape],index = ["MSE", "MAE", "R2", "MAPE"]).transpose()


if __name__ == "__main__":
    import yaml
    from yaml import SafeLoader
    from omegaconf import OmegaConf
    
    with open("time_series_template/train/tstest.yaml", "r") as f:
         cfg = yaml.load(f, Loader=SafeLoader)

    
    data = pd.read_excel("./data/time_series_train.xlsx")
    test_data = pd.read_excel("./data/time_series_test.xlsx")
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


        experiment = TimeSeriesAnomalyExperiment(cfg=cfg)
        experiment.run(data, experiment_id, run_id)
        y, metrics = experiment.predict(test_data)
        experiment.save()
    
        n_exp = experiment.load(experiment.run_id)
        
    ny, nmerics = n_exp.predict(test_data)
    print("Done!")