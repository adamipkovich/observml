import mlflow
from mlflow.tracking import MlflowClient
from framework.plugins.mlops import MLOpsPlugin
from typing import Any, Dict, Tuple
import logging
from contextlib import contextmanager

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
            self.client.search_experiments()
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
        """Start a new run, ending any existing active run first"""
        try:
            # Check for active run and end it if exists
            active_run = mlflow.active_run()
            if active_run:
                logging.info(f"Ending previous active run: {active_run.info.run_id}")
                mlflow.end_run()
            
            # Start new run
            run = mlflow.start_run(experiment_id=experiment_id)
            logging.info(f"Started new MLflow run: {run.info.run_id}")
            return run.info.run_id
        except Exception as e:
            logging.error(f"Error starting MLflow run: {e}")
            raise
    
    def log_artifact(self, path: str, artifact_path: str) -> None:
        """Log an artifact"""
        mlflow.log_artifact(path, artifact_path)
    
    def get_run(self, run_id: str) -> Any:
        """Get a run by ID"""
        return mlflow.get_run(run_id)
    
    def get_experiment_by_name(self, name: str) -> Any:
        """Get an experiment by name"""
        return self.client.get_experiment_by_name(name)
    
    def kill_run(self, run_id: str) -> bool:
        """Terminate an active run
        
        Args:
            run_id: ID of the run to terminate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First, try to end any active run in the current process
            active_run = mlflow.active_run()
            if active_run:
                if active_run.info.run_id == run_id:
                    mlflow.end_run()
                    logging.info(f"Ended active MLflow run {run_id}")
                else:
                    # End any active run, even if it's not the target run
                    mlflow.end_run()
                    logging.info(f"Ended different active run: {active_run.info.run_id}")
            
            # Then, use the client to ensure the run is terminated on the server
            try:
                self.client.set_terminated(run_id)
                logging.info(f"Terminated MLflow run {run_id} on server")
            except Exception as server_error:
                # Run might already be terminated or not exist
                logging.warning(f"Could not terminate run {run_id} on server: {server_error}")
            
            return True
        except Exception as e:
            logging.error(f"Error terminating MLflow run {run_id}: {e}")
            return False
    
    def end_any_active_run(self) -> bool:
        """End any currently active run
        
        Returns:
            True if a run was ended, False if no active run
        """
        try:
            active_run = mlflow.active_run()
            if active_run:
                mlflow.end_run()
                logging.info(f"Ended active MLflow run: {active_run.info.run_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error ending active run: {e}")
            return False
    
    @contextmanager
    def mlflow_run_context(self, experiment_id: str):
        """Context manager for MLflow runs to ensure proper cleanup
        
        Args:
            experiment_id: ID of the experiment
            
        Yields:
            run_id: ID of the created run
        """
        run_id = None
        try:
            run_id = self.start_run(experiment_id)
            yield run_id
        except Exception as e:
            logging.error(f"Error in MLflow run context: {e}")
            raise
        finally:
            if run_id:
                # Always try to end the run, even if an exception occurred
                try:
                    active_run = mlflow.active_run()
                    if active_run and active_run.info.run_id == run_id:
                        mlflow.end_run()
                        logging.info(f"Ended MLflow run in context cleanup: {run_id}")
                except Exception as cleanup_error:
                    logging.error(f"Error during run cleanup: {cleanup_error}")
