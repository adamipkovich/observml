from framework.plugins.base import Plugin
from typing import Any, Optional

class MLOpsPlugin(Plugin):
    """Plugin for MLOps operations"""
    plugin_type = "mlops"
    
    def set_tracking_uri(self, uri: str) -> None:
        """Set tracking URI"""
        ...
    
    def create_experiment(self, name: str) -> str:
        """Create a new experiment"""
        ...
    
    def start_run(self, experiment_id: str) -> str:
        """Start a new run"""
        ...
    
    def log_artifact(self, path: str, artifact_path: str) -> None:
        """Log an artifact"""
        ...
    
    def get_run(self, run_id: str) -> Any:
        """Get a run by ID"""
        ...
    
    def get_experiment_by_name(self, name: str) -> Any:
        """Get an experiment by name"""
        ...
