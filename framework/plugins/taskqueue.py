from framework.plugins.base import Plugin
from typing import Any, Optional

class TaskQueuePlugin(Plugin):
    """Plugin for task queue operations"""
    plugin_type = "taskqueue"
    
    def submit_task(self, task_name: str, *args, **kwargs) -> str:
        """Submit a task to the queue"""
        ...
    
    def get_result(self, task_id: str, timeout: Optional[int] = None) -> Any:
        """Get the result of a task"""
        ...
    
    def get_status(self, task_id: str) -> str:
        """Get the status of a task"""
        ...
