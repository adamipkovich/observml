from framework.Experiment import Experiment, load_object
import pandas as pd
import os
import logging
from asyncio import sleep
import time
import json
import yaml
from yaml import SafeLoader
from copy import deepcopy
from typing import Dict, Any, Optional, List, Type, Tuple

from framework.plugins.base import Plugin
from framework.plugins.mlops import MLOpsPlugin
from framework.plugins.datastream import DataStreamPlugin
from framework.plugins.taskqueue import TaskQueuePlugin

class ExperimentHub:
    """Class to manage multiple experiments with plugin support.
    
    This class abstracts from the user the need to manage multiple experiments.
    It uses plugins for MLOps, data streaming, and task queue operations.
    
    Attributes:
        experiments: Dictionary to store experiments.
        run_ids: Dictionary to store run IDs.
        experiment_ids: Dictionary to store experiment IDs.
        available: Dictionary to store availability of experiments.
        plugins: Dictionary to store plugins.
        task_ids: Dictionary to store task IDs.
        available_experiments: Dictionary to store available experiment types.
    """

    experiments: Dict[str, Experiment] = {}
    run_ids: Dict[str, str] = {}
    experiment_ids: Dict[str, str] = {}
    available: Dict[str, bool] = {}
    plugins: Dict[str, Plugin] = {}
    task_ids: Dict[str, str] = {}
    available_experiments: Dict[str, Dict[str, str]] = {}

    def __init__(self, **kwargs) -> None:
        """Initialize the ExperimentHub with plugins.
        
        Args:
            **kwargs: Keyword arguments for plugin initialization.
                mlflow_uri: URI for MLflow tracking server.
                rabbit_host: Hostname for RabbitMQ server.
                rabbit_port: Port for RabbitMQ server.
                rabbit_user: Username for RabbitMQ server.
                rabbit_password: Password for RabbitMQ server.
                celery_broker_url: Broker URL for Celery.
                celery_backend_url: Backend URL for Celery.
                mlops_plugin: MLOps plugin instance.
                datastream_plugin: DataStream plugin instance.
                taskqueue_plugin: TaskQueue plugin instance.
        """
        # Initialize plugins
        self._init_plugins(**kwargs)
    
    @classmethod
    def from_config(cls, config_path: str = "hub_config.yaml") -> "ExperimentHub":
        """Create an ExperimentHub instance from a configuration file
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            ExperimentHub instance
        """
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Create ExperimentHub instance
        hub = cls()
        
        # Initialize plugins from configuration
        hub._init_plugins_from_config(config["plugins"])
        
        # Register available experiment types
        hub._register_experiments(config["experiments"])
        
        return hub
    
    def _init_plugins_from_config(self, plugin_config: dict) -> None:
        """Initialize plugins from configuration
        
        Args:
            plugin_config: Plugin configuration dictionary
        """
        # Initialize MLOps plugin
        if plugin_config.get("mlops", {}).get("enabled", False):
            mlops_config = plugin_config["mlops"]
            if mlops_config["type"] == "mlflow":
                from framework.plugins.mlflow_plugin import MLflowPlugin
                mlops_plugin = MLflowPlugin(**mlops_config["config"])
                self.register_plugin(mlops_plugin)
        
        # Initialize DataStream plugin
        if plugin_config.get("datastream", {}).get("enabled", False):
            datastream_config = plugin_config["datastream"]
            if datastream_config["type"] == "rabbitmq":
                from framework.plugins.rabbitmq_plugin import RabbitMQPlugin
                datastream_plugin = RabbitMQPlugin(**datastream_config["config"])
                self.register_plugin(datastream_plugin)
        
        # Initialize TaskQueue plugin
        if plugin_config.get("taskqueue", {}).get("enabled", False):
            taskqueue_config = plugin_config["taskqueue"]
            if taskqueue_config["type"] == "celery":
                from framework.plugins.celery_plugin import CeleryPlugin
                taskqueue_plugin = CeleryPlugin(**taskqueue_config["config"])
                self.register_plugin(taskqueue_plugin)
    
    def _register_experiments(self, experiment_config: list) -> None:
        """Register available experiment types
        
        Args:
            experiment_config: List of experiment configurations
        """
        self.available_experiments = {}
        
        for exp in experiment_config:
            if exp.get("enabled", True):
                self.available_experiments[exp["name"]] = {
                    "module": exp["module"],
                    "class": exp["class"]
                }
    
    def _init_plugins(self, **kwargs) -> None:
        """Initialize plugins based on provided configuration."""
        # MLOps plugin (MLflow)
        if 'mlops_plugin' in kwargs:
            self.register_plugin(kwargs['mlops_plugin'])
        elif 'mlflow_uri' in kwargs:
            from framework.plugins.mlflow_plugin import MLflowPlugin
            mlops_plugin = MLflowPlugin(mlflow_uri=kwargs['mlflow_uri'])
            self.register_plugin(mlops_plugin)
        
        # Data stream plugin (RabbitMQ)
        if 'datastream_plugin' in kwargs:
            self.register_plugin(kwargs['datastream_plugin'])
        elif all(k in kwargs for k in ['rabbit_host', 'rabbit_port', 'rabbit_user', 'rabbit_password']):
            from framework.plugins.rabbitmq_plugin import RabbitMQPlugin
            datastream_plugin = RabbitMQPlugin(
                host=kwargs['rabbit_host'],
                port=kwargs['rabbit_port'],
                username=kwargs['rabbit_user'],
                password=kwargs['rabbit_password']
            )
            self.register_plugin(datastream_plugin)
        
        # Task queue plugin (Celery)
        if 'taskqueue_plugin' in kwargs:
            self.register_plugin(kwargs['taskqueue_plugin'])
        elif all(k in kwargs for k in ['celery_broker_url', 'celery_backend_url']):
            from framework.plugins.celery_plugin import CeleryPlugin
            taskqueue_plugin = CeleryPlugin(
                broker_url=kwargs['celery_broker_url'],
                backend_url=kwargs['celery_backend_url'],
                app_name=kwargs.get('celery_app_name', 'experiment_hub')
            )
            self.register_plugin(taskqueue_plugin)
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin.
        
        Args:
            plugin: The plugin to register.
        """
        self.plugins[plugin.plugin_type] = plugin
        plugin.initialize()
    
    def get_plugin(self, plugin_type: str) -> Optional[Plugin]:
        """Get a plugin by type.
        
        Args:
            plugin_type: The type of plugin to get.
        
        Returns:
            The plugin, or None if not found.
        """
        return self.plugins.get(plugin_type)
    
    def load(self, name: str, run_id: str = None) -> None:
        """Load a saved experiment.
        
        Args:
            name: Name of the experiment.
            run_id: Run ID of the experiment.
        """
        if name in self.run_ids.keys():
            run_id = self.run_ids[name]
        else:
            if run_id is None:
                logging.error("Run ID is not specified.")
                return "Run id was not found."
            else:
                self.run_ids[name] = run_id
        
        # Get MLOps plugin
        mlops_plugin = self.get_plugin("mlops")
        
        # Load experiment using helper function from tasks.py
        from framework.tasks import load_experiment
        experiment = load_experiment(name, run_id, mlops_plugin)
        
        self.experiments[name] = experiment
        self.available[name] = True
    
    def save(self, name: str) -> None:
        """Save an experiment.
        
        Args:
            name: Name of the experiment.
        """
        try:
            # Get MLOps plugin
            mlops_plugin = self.get_plugin("mlops")
            
            # Save experiment using helper function from tasks.py
            from framework.tasks import save_experiment
            save_experiment(self.experiments[name], self.run_ids[name], mlops_plugin)
        except Exception as e:
            logging.error(e)
    
    def kill(self, name: str, **kwargs) -> None:
        """Kill an experiment.
        
        Args:
            name: Name of the experiment.
        """
        try:
            self.experiments.pop(name)
        except Exception as e:
            logging.error(e)
    
    async def train(self, name: str, cfg: dict):
        """Train an experiment.
        
        Args:
            name: Name of the experiment.
            cfg: Configuration for the experiment.
        
        Returns:
            Task ID if using Celery, or True if trained synchronously.
        """
        self.available[name] = False
        
        # Get data stream plugin
        datastream_plugin = self.get_plugin("datastream")
        if not datastream_plugin:
            logging.error("Data stream plugin not found.")
            return
        
        # Create queue
        datastream_plugin.create_queue(name)
        
        # Get data
        data = datastream_plugin.pull_data(name)
        
        # Get MLOps plugin
        mlops_plugin = self.get_plugin("mlops")
        
        # Get task queue plugin
        taskqueue_plugin = self.get_plugin("taskqueue")
        if taskqueue_plugin:
            # Submit training task to Celery
            task_id = taskqueue_plugin.submit_task(
                "framework.tasks.train_experiment",
                name,
                cfg,
                data,
                mlops_plugin,
                datastream_plugin
            )
            
            # Store task ID
            self.task_ids[name] = task_id
            
            # Wait for result
            result_json = taskqueue_plugin.get_result(task_id)
            result = json.loads(result_json)
            
            # Update experiment info
            run_id = result["run_id"]
            experiment_id = result["experiment_id"]
            
            # Load the trained experiment
            self.load(name, run_id)
            
            self.run_ids[name] = run_id
            self.experiment_ids[name] = experiment_id
            self.available[name] = True
            
            return task_id
        else:
            # Fall back to synchronous processing using tasks.py
            from framework.tasks import train_experiment
            result_json = train_experiment(name, cfg, data, mlops_plugin, datastream_plugin)
            result = json.loads(result_json)
            
            # Update experiment info
            run_id = result["run_id"]
            experiment_id = result["experiment_id"]
            
            # Load the trained experiment
            self.load(name, run_id)
            
            self.run_ids[name] = run_id
            self.experiment_ids[name] = experiment_id
            self.available[name] = True
            
            return True
    
    async def retrain(self, name: str):
        """Retrain an experiment.
        
        Args:
            name: Name of the experiment.
        
        Returns:
            Task ID if using Celery, or True if retrained synchronously.
        """
        if self.experiments.get(name, None) is None:
            return
        
        self.available[name] = False
        run_id = self.run_ids[name]
        
        # Get MLOps plugin
        mlops_plugin = self.get_plugin("mlops")
        
        # Get task queue plugin
        taskqueue_plugin = self.get_plugin("taskqueue")
        if taskqueue_plugin:
            # Submit retraining task to Celery
            task_id = taskqueue_plugin.submit_task(
                "framework.tasks.retrain_experiment",
                name,
                run_id,
                mlops_plugin
            )
            
            # Store task ID
            self.task_ids[name] = task_id
            
            # Wait for result
            result_json = taskqueue_plugin.get_result(task_id)
            result = json.loads(result_json)
            
            # Update experiment info
            new_run_id = result["run_id"]
            experiment_id = result["experiment_id"]
            
            # Load the retrained experiment
            self.load(name, new_run_id)
            
            self.run_ids[name] = new_run_id
            self.experiment_ids[name] = experiment_id
            self.available[name] = True
            
            return task_id
        else:
            # Fall back to synchronous processing using tasks.py
            from framework.tasks import retrain_experiment
            result_json = retrain_experiment(name, run_id, mlops_plugin)
            result = json.loads(result_json)
            
            # Update experiment info
            new_run_id = result["run_id"]
            experiment_id = result["experiment_id"]
            
            # Load the retrained experiment
            self.load(name, new_run_id)
            
            self.run_ids[name] = new_run_id
            self.experiment_ids[name] = experiment_id
            self.available[name] = True
            
            return True
    
    def plot(self, name: str, plot_name: str):
        """Get a plot from an experiment.
        
        Args:
            name: Name of the experiment.
            plot_name: Name of the plot.
        
        Returns:
            The plot, or an error message if not found.
        """
        if name not in self.experiments.keys() or not self.available[name] or name == 'None' or plot_name == 'None':
            return "Experiment not found."
        
        return self.experiments[name]._report_registry[plot_name]
    
    def plot_names(self, name: str) -> List[str] | str:
        """Get plot names from an experiment.
        
        Args:
            name: Name of the experiment.
        
        Returns:
            List of plot names, or an error message if not found.
        """
        if self.experiments and name != 'None' and self.experiments[name]:
            return [list(self.experiments[name]._report_registry.keys()), list(self.experiments[name]._eda_registry.keys())]
        else:
            return "No reports available."
    
    async def predict(self, name: str) -> pd.DataFrame:
        """Make predictions with an experiment.
        
        Args:
            name: Name of the experiment.
        
        Returns:
            Predictions from the experiment.
        """
        available = self.available[name]
        while not available:
            await sleep(0.2)
            available = self.available[name]
            logging.info(f"Waiting for {name} to be available.")
        
        # Get data stream plugin
        datastream_plugin = self.get_plugin("datastream")
        if not datastream_plugin:
            logging.error("Data stream plugin not found.")
            return
        
        # Get data
        data_json = datastream_plugin.pull_data(name)
        
        # Get MLOps plugin
        mlops_plugin = self.get_plugin("mlops")
        
        # Get task queue plugin
        taskqueue_plugin = self.get_plugin("taskqueue")
        if taskqueue_plugin:
            # Submit prediction task to Celery
            task_id = taskqueue_plugin.submit_task(
                "framework.tasks.predict_experiment",
                name,
                self.run_ids[name],
                data_json,
                mlops_plugin
            )
            
            # Wait for result
            result_json = taskqueue_plugin.get_result(task_id)
            result = json.loads(result_json)
            
            # Parse predictions
            predictions = pd.read_json(result["predictions"])
            
            # Check if retraining is needed
            if result["needs_retrain"]:
                await self.retrain(name)
            
            return predictions
        else:
            # Fall back to synchronous processing
            experiment = self.experiments[name]
            data = pd.read_json(data_json)
            predictions = experiment.predict(data)
            
            # Check if retraining is needed
            if experiment.retrain():
                await self.retrain(name)
            
            return predictions
    
    def get_train_data(self, name: str) -> str:
        """Get training data from an experiment.
        
        Args:
            name: Name of the experiment.
        
        Returns:
            Training data in JSON format.
        """
        if name not in self.experiments.keys() or not self.available[name]:
            return "Experiment not found."
        return self.experiments[name].data.reset_index(inplace=False).to_json()
    
    def get_cfg(self, name: str):
        """Get configuration of an experiment.
        
        Args:
            name: Name of the experiment.
        
        Returns:
            Configuration of the experiment.
        """
        if name not in self.experiments.keys() or not self.available[name]:
            logging.warning(f"Experiment {name} not found.")
            return {}
        return self.experiments[name].cfg
    
    def flush(self, queue: str) -> None:
        """Flush a queue.
        
        Args:
            queue: Name of the queue.
        """
        # Get data stream plugin
        datastream_plugin = self.get_plugin("datastream")
        if datastream_plugin:
            datastream_plugin.flush_queue(queue)
    
    def plot_eda(self, name: str, fig_name: str):
        """Get an EDA figure from an experiment.
        
        Args:
            name: Name of the experiment.
            fig_name: Name of the figure.
        
        Returns:
            The EDA figure, or an error message if not found.
        """
        if name not in self.experiments.keys() or not self.available[name] or name == 'None' or fig_name == 'None':
            return "Experiment not found."
        
        return self.experiments[name]._eda_registry[fig_name]
    
    def check_plugin_health(self) -> Dict[str, Dict[str, Any]]:
        """Check the health of all registered plugins
        
        Returns:
            Dictionary with plugin health information
        """
        health_info = {}
        
        for plugin_type, plugin in self.plugins.items():
            is_healthy, details = plugin.health_check()
            health_info[plugin_type] = {
                "healthy": is_healthy,
                "details": details
            }
        
        return health_info
    
    def create_experiment(self, name: str, experiment_type: str, config: dict) -> str:
        """Create a new experiment
        
        Args:
            name: Name of the experiment
            experiment_type: Type of experiment (must be registered)
            config: Configuration for the experiment
            
        Returns:
            Name of the created experiment
        """
        if experiment_type not in self.available_experiments:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        exp_info = self.available_experiments[experiment_type]
        
        # Add load_object information to config
        config["load_object"] = {
            "module": exp_info["module"],
            "name": exp_info["class"]
        }
        
        # Train the experiment
        self.train(name, config)
        
        return name
