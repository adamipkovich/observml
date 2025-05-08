import mlflow
from mlflow.tracking import MlflowClient
from framework.plugins.mlops import MLOpsPlugin
from typing import Any, Dict, Tuple

class MLflowPlugin(MLOpsPlugin):
    """MLflow implementation of MLOpsPlugin"""
    plugin_type = "mlops"
    
    def __init__(self, mlflow_uri: str, **kwargs):
        self.mlflow_uri = mlflow_uri
        self.client = None
    
    def initialize(self) -> None:
        """Initialize the plugin"""
        self.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient(self.mlflow_uri)
    
    def shutdown(self) -> None:
        """Clean up resources"""
        # Nothing specific to clean up for MLflow
        pass
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if MLflow plugin is working correctly"""
        try:
            # Try to connect to MLflow server
            self.client.list_experiments()
            return True, {"status": "connected", "uri": self.mlflow_uri}
        except Exception as e:
            return False, {"status": "error", "message": str(e), "uri": self.mlflow_uri}
    
    def set_tracking_uri(self, uri: str) -> None:
        """Set MLflow tracking URI"""
        mlflow.set_tracking_uri(uri)
    
    def create_experiment(self, name: str) -> str:
        """Create a new experiment"""
        try:
            return self.client.create_experiment(name)
        except Exception:
            # Experiment might already exist
            return self.get_experiment_by_name(name).experiment_id
    
    def start_run(self, experiment_id: str) -> str:
        """Start a new run"""
        run = mlflow.start_run(experiment_id=experiment_id)
        return run.info.run_id
    
    def log_artifact(self, path: str, artifact_path: str) -> None:
        """Log an artifact"""
        mlflow.log_artifact(path, artifact_path)
    
    def get_run(self, run_id: str) -> Any:
        """Get a run by ID"""
        return mlflow.get_run(run_id)
    
    def get_experiment_by_name(self, name: str) -> Any:
        """Get an experiment by name"""
        return self.client.get_experiment_by_name(name)
