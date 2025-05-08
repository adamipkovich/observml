from __future__ import annotations
import os
from typing import Protocol
import mlflow
import numpy as np
import pandas as pd
import joblib
import logging
from mlflow.tracking import MlflowClient
from copy import deepcopy
from plotly import graph_objects as go
import pickle
from importlib import import_module
import json
import yaml
from plotly.io import from_json, to_json, write_json, read_json
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
import tensorflow as tf


class Experiment(Protocol):
    """A class representing an experiment. This class is a protocol that defines the methods that an experiment should implement.
    Some functions must be overloaded by an implementation class, while others can be left as is. Can only handle sklearn.BaseEstimator class, and so the used model must be overloaded
    
    Parameters:
        mlflow_uri (str): the URI of the MLflow server
        model (any): the model to be used in the experiment
        data (pd.DataFrame): the training data -- allocated during training
        new_data (pd.DataFrame): data that is not in the training set -- incoming data in prediction
        prediction (pd.DataFrame): the prediction of the new data
        metrics (pd.DataFrame): the metrics of the model
        cfg (dict): the configuration of the experiment
        _model_registry (dict[str, type]): a registry of model classes
        _report_registry (dict[str, go.Figure]): a registry of reports
        run_id (str): the ID of the run
        experiment_id (str): the ID of the experiment
        name (str): the name of the experiment
        eda_report (str): the EDA report


    
    TODO:
    - proper logging
    - Add mmlw_estimator as a base class for the model
    - SPC chart
    - add support for data drift (eda)
    - add support for model drift (eda)
    - add support for model explainability (shap, lime, etc.) 
    - add support for general data preprocessing (e.g. missing values, outliers, etc.)
    - add support for dim reduction (PCA, TSNE, etc.)
    - add support for feature selection (RFE, etc.)
    - dynamic model loading from specific folder/registry
    """

    mlflow_uri : str = None
    model : any = None
    data : pd.DataFrame = None
    new_data : pd.DataFrame = pd.DataFrame([])
    metrics : pd.DataFrame = None
    cfg : dict = None
    _model_registry : dict[str, type] = dict()
    _report_registry : dict[str, go.Figure] = dict()
    run_id : str = None
    experiment_id : str = None
    name : str = ""
    _eda_registry : dict[str, go.Figure] = dict()
    scheme = None
    ###retrain_window : int, metric : str, metric_threshold : float, higher_better : bool = False
    retrain_window : int = 0
    metric : str = None
    metric_threshold : float = 0.0
    higher_better : bool = False
    data_format : dict = None
    
    def __init__(self, cfg : dict, experiment_id:str, run_id:str) -> None:
        """Initialize the Experiment class. This class is a protocol that defines the methods that an experiment should implement.

        Parameters:
            cfg (dict): the configuration of the experiment
            experiment_id (str): the ID of the experiment
            run_id (str): the ID of the run
        """

        self.cfg = cfg
        self.experiment_id = experiment_id
        self.run_id = run_id

    def format_data(self, data, format : dict = None) -> pd.DataFrame:
        """This function will provide a function to recover data from nonstandard json as pd.Dataframe.
        Parameters:
            data : data in json format or DataFrame
            format (dict) : settings for formatting. If None, then default pd.read_json is used."""

        if format is None:
            if isinstance(data, pd.DataFrame):
                return data
            else:
                from io import StringIO
                data = pd.read_json(StringIO(data))
                return data
        else:
            if format.get("name", "pivot") == "pivot":
                data = json.loads(data)
                _id = format.get("id", "tsdata")
                mxlvl = format.get("max_level", 1)
                data = pd.json_normalize(data[_id], max_level = 1)
                column = format.get("columns", "target")
                ind = format.get("index", "date")
                vals = format.get("values", "value") 
                data = data.pivot(columns=column, index= ind, values=vals)
                data.columns.name = None
                data.reset_index(drop = False, inplace = True)
                return data
            else:
                raise Exception(f"Configuration contains faulty data formatting settings. Please make sure the setup keyword contaisn the necessary information. Settings are: {format}")

    ###ADD Overloadable methods for Experiment class.
    def setup(self, data:pd.DataFrame, *args, **kwargs):
        """This function sets up the experiment. It is called before training the model."""
        return NotImplementedError("Implement this in the child class.")
    
    def create_model(self, *args, **kwargs):
        """This function trains the model."""
        return NotImplementedError("Implement this in the child class.")
    
    def predict(self, data, *args, **kwargs):
        """This function predicts the target variable."""
        return NotImplementedError("Implement this in the child class.")

    def _score(self, y, y_hat) -> pd.DataFrame:
        """This function calculates the metrics of the model."""
        return NotImplementedError("Implement this in the child class.")
    
    def plot_model(self, plot:str) -> go.Figure:
        """This function plots the model.

        Parameters:
            plot (str): the plot name that is to be displayed

        Returns:
            go.Figure: the plot
        """
        if plot not in self.model._figs:
            raise ValueError(f"Plot {plot} not found in model.")
        return self.model._figs[plot]
    
    def save(self):
        """This function saves the experiment. It saves the model and the reports. Uses joblib to save the model and pickle to save the reports.
        """
        with mlflow.start_run(nested = True, experiment_id=self.experiment_id, run_id=self.run_id) as run:  
            
            repository = get_artifact_repository(run.info.artifact_uri)
            try:
                repository.delete_artifacts(mlflow.get_artifact_uri("experiment.pkl"))
                repository.delete_artifacts(mlflow.get_artifact_uri("metadata.yaml"))
                repository.delete_artifacts(mlflow.get_artifact_uri("reports"))
            except:
                pass
        
        experiment_to_be_saved = deepcopy(self)
        if isinstance(self.model, tf.keras.models.Model):
            self.model.save(os.path.join(os.getcwd(), "runs", self.run_id, "model.keras"))
            experiment_to_be_saved.model = None
            with mlflow.start_run(nested = True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
                mlflow.log_artifact(os.path.join(os.getcwd(),"runs", self.run_id, "model.keras"), "model")


        with open(os.path.join(os.getcwd(), "runs", self.run_id, "experiment.pkl"), "wb") as f:
            joblib.dump(experiment_to_be_saved, f)
 
        for k, v in self._report_registry.items():  
            
            with open(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"{k}.json"), "w") as f:
                write_json(v, f)  #os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"{k}.json") 

            with mlflow.start_run(nested = True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
                mlflow.log_artifact(os.path.join(os.getcwd(),"runs", self.run_id,"reports" , f"{k}.json"), "reports")

        for k,v in self._eda_registry.items():
            with open(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"eda_{k}.json"), "w") as f:
                write_json(v, f)
            with mlflow.start_run(nested = True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
                mlflow.log_artifact(os.path.join(os.getcwd(),"runs", self.run_id,"reports" , f"eda_{k}.json"), "reports")

        yaml.dump(self.cfg, open(os.path.join(os.getcwd(), "runs", self.run_id, "metadata.yaml"), "w"))

        with mlflow.start_run(nested = True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
                mlflow.log_artifact(os.path.join(os.getcwd(),"runs", self.run_id,  "experiment.pkl"), "")
                mlflow.log_artifact(os.path.join(os.getcwd(),"runs", self.run_id, "metadata.yaml"), "")


    def load(self, run_id:str) -> Experiment:
        """This function loads the experiment. It loads the model and the reports. Uses joblib to load the model and pickle to load the reports.
        
        Parameters:
            run_id (str): the ID of the run

        Returns:
            Experiment: the experiment
        """
        dst = os.path.join(os.getcwd(), "runs", run_id)
        mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/experiment.pkl", dst_path= dst)
        exp = joblib.load(os.path.join(dst, "experiment.pkl")) 

        if os.path.exists(os.path.join(dst, "model.keras")):
            exp.model = tf.keras.models.load_model(os.path.join(dst, "model.keras"))

        mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/reports", dst_path= dst)
        report_dir = os.path.join(dst, "reports")
        for file in os.listdir(report_dir):
            if file.endswith(".json") and file.startswith("eda_"):
                with open(os.path.join(report_dir, file), "rb") as f:
                        report = read_json(f)
                        key = file[4:].split(".")[0]
                        exp._eda_registry[key] = report  
 
            elif file.endswith(".json"):
                with open(os.path.join(report_dir, file), "rb") as f:
                     report = read_json(f)
                     exp._report_registry[file.split(".")[0]] = report

        ##find .pkl files in report  

        exp.model._figs = exp._report_registry
        exp.cfg = self.cfg
        exp.experiment_id = self.experiment_id
        exp.run_id = run_id

        ## load _report_registry
        ## assing it to model

        #exp.model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        return exp
    
    def convert_datetime(self, data : pd.DataFrame, format : str):
        # TODO: get and return column only...
        if np.issubdtype(data[self.ds].dtype, np.integer):
            data[self.ds] = pd.to_datetime(data[self.ds], unit = format)#, format = self.format .dt.strftime('%Y-%m-%d %H:%M:%S. %s') .astype(str), format = self.format
        elif data[self.ds].dtype == str:
            data[self.ds] = pd.to_datetime(data[self.ds], format = format)#, format = self.format .dt.strftime('%Y-%m-%d %H:%M:%S. %s') .astype(str), format = self.format
        else:
            logging.warn(f"Unknown type {data[self.ds].dtype}")    
            data[self.ds] = pd.to_datetime(data[self.ds])
        return data

    def eda(self): # interactions = None,
        """This function performs simple exploratory data analysis.
        
        Returns:
            None
        """
        print("Performing EDA...")
        ## add histograms
        self._eda_registry = dict()
        
        for col in self.data.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=self.data[col], name=col))
            fig.update_layout(title_text=f"{col} Histogram")
            fig.update_xaxes(title_text=col)
            fig.update_yaxes(title_text="Count")
            self._eda_registry[col] = fig
        
        ## add correlation matrix
        labels = self.data.columns.to_list()
        labels.reverse()
        fig = go.Figure(data=go.Heatmap(z=self.data.corr(), x=labels, y=labels))
        self._eda_registry["correlation_matrix"] = fig    

        ## add missing values
        #fig = go.Figure(data=go.Heatmap(z=self.data.isnull().sum(), x=self.data.columns, y=self.data.columns))
        #self._eda_registry["missing_values"] = fig

    
    def retrain(self):
        if self.metric is not None:
            if self.higher_better:
               print(self.metrics.iloc[-1].loc[self.metric])
               
               return  self.metrics.iloc[-1].loc[self.metric] < self.metric_threshold
            return self.metrics.iloc[-1].loc[self.metric]> self.metric_threshold
            
        return False
    
    def spc(self):
        """This function creates a statistical process control chart. WIP."""
        pass

    def run(self, data:pd.DataFrame) -> None:
        """This function runs the experiment. It trains the model and performs exploratory data analysis. Handles everythin internally...
        
        Parameters:
            data (pd.DataFrame): the training data
        
        """

        if data is None:
            raise ValueError("Data not found. Please provide data for training.")

        if self.cfg is None:
            raise ValueError("No training configuration found in metadata. The interface is not properly configured.")
        
        ## for k in config run the function, with specified parameters...
        funcs = list(self.cfg.keys())
        funcs.remove("load_object")

        self.data = data
        for k in funcs:
            if k == "setup":
                self.setup(data)
                #self.eda(   )
            else:
                getattr(self, k)()
        
        
        #self.setup(data, experiment_id, run_id)
        #self.model = self.create_model()
        #self.eda() # -> add eda to figs
        #self.spc() # -> add spc to figs
        #self.spc_chart()
        return None
    
    def join_data(self):
        """Joins new data and previous train data."""
        ndata = pd.concat((self.data, self.new_data), axis=0)
        if self.retrain_window != 0 and self.retrain_window < ndata.shape[0]:
            ndata = ndata.iloc[-self.retrain_window:, :]
            
        
        return ndata

    def export(self) -> None:
        """This function exports the reports as HTML files and logs them to MLflow."""
        with mlflow.start_run(nested=True, experiment_id=self.experiment_id, run_id=self.run_id) as run:
            for k, v in self._eda_registry.items():
                v.write_html(os.path.join(os.getcwd(),"runs", self.run_id, "reports", f"eda_{k}_report.html"))
                mlflow.log_artifact(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"eda_{k}_report.html"), "reports")
            
            for k, v in self.model._figs.items(): 
                v.write_html(os.path.join(os.getcwd(),"runs", self.run_id, "reports", f"{k}_report.html"))
                mlflow.log_artifact(os.path.join(os.getcwd(), "runs", self.run_id, "reports", f"{k}_report.html"), "reports")

    def get_eda_reports(self)   -> list[str]:
        """This function returns the IDs of the EDA reports that are available in the model.
        
        Returns:
            list[str]: the IDs of the reports
        """
        return list(self._eda_registry.keys())       

    def get_fig_types(self) -> list[str]:
        """This function returns the IDs of figures that are available in the model.
        
        Returns:
            list[str]: the IDs of figures
        """
        return list(self.model._figs.keys())
    

def load_object(module : str, name: str) -> any:
        """This function loads an object/function from a module.
        
        Parameters:

            module (str): the module to load the object from (e.g. sklearn.preprocessing)
            name (str): the name of the object (e.g. StandardScaler)
        
        Returns:
            any: the class/function
        
        Example:
            for a class:
                >>> class = load_object("sklearn.preprocessing", "StandardScaler")
                >>> obj = class()

            for a function:
                >>> func = load_object("sklearn.preprocessing", "scale")
                >>> func([1,2,3])
        """
        m = import_module(module)
        return getattr(m, name)


    # def spc_chart(self, update_fig = False):
#         """Create a card for the SPC report."""
#         data = self.train_data
#         if update_fig:
#             fig = self.exp.model._figs["spc"]
#         else:
#             fig = make_subplots(rows = 1, cols = 1)
#         col = self.target
#         mean = np.mean(self.train_data.loc[:, col])
#         std = np.std(self.train_data.loc[:, col])
#         cond = np.logical_and(data.loc[:, col] < mean + std, data.loc[:, col] >  mean - std)
#         subfig = go.Scatter(x = data.index.to_series().loc[cond].reset_index(drop = True), y =data.loc[cond, col].reset_index(drop=True), name = "Inline", mode = "markers", marker_color = "blue", 
#             marker=dict(
#                 symbol="circle", #"circle-x"
#                 size=3), )
#         fig.add_trace(subfig, row = 1, col = 1)
#         fig.update_yaxes(title_text=col, row=1, col=1)
#         if not update_fig:
#                 fig.add_hline(y = mean, line_color="green", row=1, col=1, annotation_text="mean")
#                 fig.add_hline(y = mean + std ,line_color="yellow", row=1, col=1,annotation_text=r"$+\sigma$")
#                 fig.add_hline(y = mean - std ,line_color="yellow", row=1, col=1,annotation_text=r"$-\sigma$")

#                 fig.add_hline(y = mean + 2*std ,line_color="orange", row=1, col=1, annotation_text=r"$+2\sigma$")
#                 fig.add_hline(y = mean - 2*std , line_color="orange", row=1, col=1, annotation_text=r"$-2\sigma$")

#                 fig.add_hline(y = mean + 3*std ,line_color="red", row=1, col=1, annotation_text=r"$+3\sigma$")
#                 fig.add_hline(y = mean - 3*std ,line_color="red", row=1, col=1, annotation_text=r"$-3\sigma$")

#         a = np.logical_and(data.loc[:, col] >  mean + std,  data.loc[:, col] <  mean + 2*std)
#         b = np.logical_and(data.loc[:, col] <  mean - std,  data.loc[:, col] >  mean - 2*std)
#         cond = np.logical_xor(a, b)
#         scatterfig = go.Scatter(x =data.index.to_series().loc[cond].reset_index(drop = True), y =data.loc[cond, col].reset_index(drop=True), mode = "markers", marker_color = "yellow", marker=dict(
#                     symbol="circle-x", size=3,), name = r"$1\sigma$ Anomalies")
#         fig.add_trace(scatterfig, row=1, col=1)

#             #2 sigma
#         a = np.logical_and(data.loc[:, col] >  mean + 2*std,  data.loc[:, col] <  mean + 3*std)
#         b = np.logical_and(data.loc[:, col] <  mean - 2*std,  data.loc[:, col] >  mean - 3*std)
#         cond = np.logical_xor(a, b)
#         scatterfig = go.Scatter(x = data.index.to_series().loc[cond].reset_index(drop = True), y =data.loc[cond, col].reset_index(drop=True), mode = "markers", marker_color = "orange",marker=dict(
#                     symbol="circle-x", size=3,), name = r"$2\sigma$ Anomalies")
#         fig.add_trace(scatterfig, row=1, col=1)

#             #3 sigma
#         cond = np.logical_xor(data.loc[:, col] >  mean + 3*std,  data.loc[:, col] < mean - 3*std)
#         scatterfig = go.Scatter(x = data.index.to_series().loc[cond].reset_index(drop = True), y =data.loc[cond, col].reset_index(drop=True), mode = "markers", marker_color = "red", marker=dict(
#                     symbol="x", size=3,), name = r"$3\sigma$ Anomalies")
#         fig.add_trace(scatterfig, row=1, col=1)
#         if not update_fig:
#             self.exp.model._figs["spc"] = fig
