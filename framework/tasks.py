from celery import shared_task
import pandas as pd
import json
import os
import yaml
from yaml import SafeLoader
from framework.Experiment import load_object
import logging
import joblib
from copy import deepcopy
import uuid
from typing import Optional

from framework.plugins.mlops import MLOpsPlugin
from framework.plugins.datastream import DataStreamPlugin

@shared_task
def train_experiment(name: str, cfg: dict, data_json: str, 
                     mlops_plugin: Optional[MLOpsPlugin] = None,
                     datastream_plugin: Optional[DataStreamPlugin] = None) -> str:
    """Train an experiment with dependency injection for plugins
    
    Args:
        name: Name of the experiment
        cfg: Configuration dictionary
        data_json: JSON string containing the data
        mlops_plugin: Optional MLOps plugin instance
        datastream_plugin: Optional DataStream plugin instance
    """
    # Parse data
    data = pd.read_json(data_json)
    
    # Get experiment class
    interface = cfg["load_object"]["name"]
    module = cfg["load_object"]["module"]
    experiment_class = load_object(module, interface)
    
    # Create experiment
    if mlops_plugin:
        # Use MLOps plugin for experiment creation and run tracking
        experiment_id = mlops_plugin.create_experiment(name)
        run_id = mlops_plugin.start_run(experiment_id)
    else:
        # Local experiment tracking
        run_id = str(uuid.uuid4())
        experiment_id = name
    
    # Create experiment instance
    experiment = experiment_class(cfg=cfg, run_id=run_id, experiment_id=experiment_id)
    
    # Create directories
    for path in [
        os.path.join(os.getcwd(), "runs"),
        os.path.join(os.getcwd(), "runs", run_id),
        os.path.join(os.getcwd(), "runs", run_id, "reports"),
        os.path.join(os.getcwd(), "runs", run_id, "model")
    ]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Run experiment
    experiment.run(data)
    
    # Save experiment
    save_experiment(experiment, run_id, mlops_plugin)
    
    return json.dumps({
        "run_id": run_id,
        "experiment_id": experiment_id
    })

@shared_task
def predict_experiment(name: str, run_id: str, data_json: str, 
                       mlops_plugin: Optional[MLOpsPlugin] = None) -> str:
    """Make predictions with dependency injection for plugins
    
    Args:
        name: Name of the experiment
        run_id: Run ID of the experiment
        data_json: JSON string containing the data
        mlops_plugin: Optional MLOps plugin instance
    """
    # Load experiment
    experiment = load_experiment(name, run_id, mlops_plugin)
    
    # Parse data
    data = pd.read_json(data_json)
    
    # Make predictions
    predictions = experiment.predict(data)
    
    # Check if retraining is needed
    needs_retrain = experiment.retrain()
    
    return json.dumps({
        "predictions": predictions.to_json(),
        "needs_retrain": needs_retrain
    })

@shared_task
def retrain_experiment(name: str, run_id: str, 
                       mlops_plugin: Optional[MLOpsPlugin] = None) -> str:
    """Retrain an experiment with dependency injection for plugins
    
    Args:
        name: Name of the experiment
        run_id: Run ID of the experiment
        mlops_plugin: Optional MLOps plugin instance
    """
    # Load experiment
    experiment = load_experiment(name, run_id, mlops_plugin)
    
    # Get data for retraining
    data = experiment.join_data()
    
    # Get experiment ID
    exp_id = experiment.experiment_id
    
    # Create new run
    if mlops_plugin:
        # Use MLOps plugin to create a new run
        new_run_id = mlops_plugin.start_run(exp_id)
    else:
        # Generate a local run ID
        new_run_id = str(uuid.uuid4())
    
    # Create new experiment
    interface_class = experiment.__class__
    new_experiment = interface_class(cfg=experiment.cfg, run_id=new_run_id, experiment_id=exp_id)
    
    # Create directories
    for path in [
        os.path.join(os.getcwd(), "runs"),
        os.path.join(os.getcwd(), "runs", new_run_id),
        os.path.join(os.getcwd(), "runs", new_run_id, "reports"),
        os.path.join(os.getcwd(), "runs", new_run_id, "model")
    ]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Run experiment
    new_experiment.run(data)
    
    # Save experiment
    save_experiment(new_experiment, new_run_id, mlops_plugin)
    
    return json.dumps({
        "run_id": new_run_id,
        "experiment_id": exp_id
    })

def load_experiment(name: str, run_id: str, mlops_plugin: Optional[MLOpsPlugin] = None):
    """Helper function to load an experiment
    
    Args:
        name: Name of the experiment
        run_id: Run ID of the experiment
        mlops_plugin: Optional MLOps plugin instance
    
    Returns:
        The loaded experiment
    """
    # Ensure run directory exists
    if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id)):
        os.makedirs(os.path.join(os.getcwd(), "runs", run_id))
        
        # Download artifacts
        if mlops_plugin:
            # Use MLOps plugin to download artifacts
            import mlflow
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/metadata.yaml", 
                dst_path=os.path.join(os.getcwd(), "runs", run_id)
            )
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/experiment.pkl", 
                dst_path=os.path.join(os.getcwd(), "runs", run_id)
            )
        else:
            # If no MLOps plugin, we assume the files are already local
            if not os.path.exists(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml")):
                raise FileNotFoundError(f"Metadata file not found for run {run_id}")
    
    # Load metadata
    with open(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml")) as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    
    # Get experiment class
    interface = cfg["load_object"]["name"]
    module = cfg["load_object"]["module"]
    interface_class = load_object(module, interface)
    
    # Get experiment ID
    if mlops_plugin:
        run = mlops_plugin.get_run(run_id)
        exp_id = run.info.experiment_id
    else:
        exp_id = name
    
    # Create experiment instance
    experiment = interface_class(cfg=cfg, run_id=run_id, experiment_id=exp_id)
    
    # Load the experiment
    experiment = experiment.load(run_id=run_id)
    
    return experiment

def save_experiment(experiment, run_id: str, mlops_plugin: Optional[MLOpsPlugin] = None):
    """Helper function to save an experiment
    
    Args:
        experiment: The experiment to save
        run_id: Run ID of the experiment
        mlops_plugin: Optional MLOps plugin instance
    """
    if mlops_plugin:
        # Use the experiment's save method which uses MLflow
        experiment.save()
    else:
        # Save locally without MLflow
        experiment_to_be_saved = deepcopy(experiment)
        
        # Save the model if it's a TensorFlow model
        import tensorflow as tf
        if hasattr(experiment, 'model') and isinstance(experiment.model, tf.keras.models.Model):
            experiment.model.save(os.path.join(os.getcwd(), "runs", run_id, "model.keras"))
            experiment_to_be_saved.model = None
        
        # Save the experiment
        with open(os.path.join(os.getcwd(), "runs", run_id, "experiment.pkl"), "wb") as f:
            joblib.dump(experiment_to_be_saved, f)
        
        # Save the reports
        from plotly.io import write_json
        for k, v in experiment._report_registry.items():
            with open(os.path.join(os.getcwd(), "runs", run_id, "reports", f"{k}.json"), "w") as f:
                write_json(v, f)
        
        # Save the EDA reports
        for k, v in experiment._eda_registry.items():
            with open(os.path.join(os.getcwd(), "runs", run_id, "reports", f"eda_{k}.json"), "w") as f:
                write_json(v, f)
        
        # Save the configuration
        yaml.dump(experiment.cfg, open(os.path.join(os.getcwd(), "runs", run_id, "metadata.yaml"), "w"))
