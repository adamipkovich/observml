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

def train_experiment(name: str, cfg: dict, data_json: str, 
                     mlops_config: Optional[dict] = None,
                     datastream_config: Optional[dict] = None) -> str:
    """Train an experiment with dependency injection for plugins
    
    Args:
        name: Name of the experiment
        cfg: Configuration dictionary
        data_json: JSON string containing the data
        mlops_config: Optional MLOps plugin configuration dictionary
        datastream_config: Optional DataStream plugin configuration dictionary
    """
    # Create plugins from configuration
    mlops_plugin = None
    if mlops_config:
        if mlops_config.get("type") == "mlflow":
            from framework.plugins.mlflow_plugin import MLflowPlugin
            mlops_plugin = MLflowPlugin(**mlops_config.get("config", {}))
            mlops_plugin.initialize()
    
    datastream_plugin = None
    if datastream_config:
        if datastream_config.get("type") == "rabbitmq":
            from framework.plugins.rabbitmq_plugin import RabbitMQPlugin
            datastream_plugin = RabbitMQPlugin(**datastream_config.get("config", {}))
            datastream_plugin.initialize()
    
    # Parse data
    from io import StringIO
    data = pd.read_json(StringIO(data_json))
    
    # Get experiment class
    interface = cfg["load_object"]["name"]
    module = cfg["load_object"]["module"]
    experiment_class = load_object(module, interface)
    
    run_id = None
    experiment_id = None
    
    try:
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
        
    except Exception as e:
        logging.error(f"Error during training experiment {name}: {e}")
        # Clean up MLflow run if it was started
        if mlops_plugin and run_id:
            try:
                mlops_plugin.kill_run(run_id)
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up failed run {run_id}: {cleanup_error}")
        raise

def predict_experiment(name: str, run_id: str, data_json: str, 
                       mlops_config: Optional[dict] = None) -> str:
    """Make predictions with dependency injection for plugins
    
    Args:
        name: Name of the experiment
        run_id: Run ID of the experiment
        data_json: JSON string containing the data
        mlops_config: Optional MLOps plugin configuration dictionary
    """
    # Create plugin from configuration
    mlops_plugin = None
    if mlops_config:
        if mlops_config.get("type") == "mlflow":
            from framework.plugins.mlflow_plugin import MLflowPlugin
            mlops_plugin = MLflowPlugin(**mlops_config.get("config", {}))
            mlops_plugin.initialize()
    # Load experiment
    experiment = load_experiment(name, run_id, mlops_plugin)
    
    # Parse data
    from io import StringIO
    data = pd.read_json(StringIO(data_json))
    
    # Make predictions
    predictions = experiment.predict(data)
    
    # Check if retraining is needed
    needs_retrain = experiment.retrain()
    
    return json.dumps({
        "predictions": predictions.to_json(),
        "needs_retrain": needs_retrain
    })

def retrain_experiment(name: str, run_id: str, 
                       mlops_config: Optional[dict] = None) -> str:
    """Retrain an experiment with dependency injection for plugins
    
    Args:
        name: Name of the experiment
        run_id: Run ID of the experiment
        mlops_config: Optional MLOps plugin configuration dictionary
    """
    # Create plugin from configuration
    mlops_plugin = None
    if mlops_config:
        if mlops_config.get("type") == "mlflow":
            from framework.plugins.mlflow_plugin import MLflowPlugin
            mlops_plugin = MLflowPlugin(**mlops_config.get("config", {}))
            mlops_plugin.initialize()
    
    # Load experiment
    experiment = load_experiment(name, run_id, mlops_plugin)
    
    # Get data for retraining
    data = experiment.join_data()
    
    # Get experiment ID
    exp_id = experiment.experiment_id
    
    new_run_id = None
    
    try:
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
        
    except Exception as e:
        logging.error(f"Error during retraining experiment {name}: {e}")
        # Clean up MLflow run if it was started
        if mlops_plugin and new_run_id:
            try:
                mlops_plugin.kill_run(new_run_id)
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up failed retrain run {new_run_id}: {cleanup_error}")
        raise

def load_experiment(name: str, run_id: str, mlops_plugin: Optional[MLOpsPlugin] = None, mlops_config: Optional[dict] = None):
    """Helper function to load an experiment
    
    Args:
        name: Name of the experiment
        run_id: Run ID of the experiment
        mlops_plugin: Optional MLOps plugin instance
        mlops_config: Optional MLOps plugin configuration dictionary
    
    Returns:
        The loaded experiment
    """
    # Create plugin from configuration if not provided directly
    if mlops_config and not mlops_plugin:
        if mlops_config.get("type") == "mlflow":
            from framework.plugins.mlflow_plugin import MLflowPlugin
            mlops_plugin = MLflowPlugin(**mlops_config.get("config", {}))
            mlops_plugin.initialize()
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

def save_experiment(experiment, run_id: str, mlops_plugin: Optional[MLOpsPlugin] = None, mlops_config: Optional[dict] = None):
    """Helper function to save an experiment
    
    Args:
        experiment: The experiment to save
        run_id: Run ID of the experiment
        mlops_plugin: Optional MLOps plugin instance
        mlops_config: Optional MLOps plugin configuration dictionary
    """
    # Create plugin from configuration if not provided directly
    if mlops_config and not mlops_plugin:
        if mlops_config.get("type") == "mlflow":
            from framework.plugins.mlflow_plugin import MLflowPlugin
            mlops_plugin = MLflowPlugin(**mlops_config.get("config", {}))
            mlops_plugin.initialize()
    if mlops_plugin:
        # Use the experiment's save method which uses MLflow
        experiment.save()
    else:
        # Save locally without MLflow
        experiment_to_be_saved = deepcopy(experiment)
        
        # Save the model if it's a TensorFlow model
        try:
            import tensorflow as tf
            # Check if the model is a TensorFlow/Keras model
            if hasattr(experiment, 'model') and hasattr(tf, 'keras') and isinstance(experiment.model, tf.keras.models.Model):
                experiment.model.save(os.path.join(os.getcwd(), "runs", run_id, "model.keras"))
                experiment_to_be_saved.model = None
        except (ImportError, AttributeError):
            # TensorFlow not available or keras not accessible, skip TensorFlow model handling
            pass
        
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
