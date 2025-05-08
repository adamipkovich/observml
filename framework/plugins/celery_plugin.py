from celery import Celery
from framework.plugins.taskqueue import TaskQueuePlugin
from typing import Any, Optional, Dict, Tuple

class CeleryPlugin(TaskQueuePlugin):
    """Celery implementation of TaskQueuePlugin"""
    plugin_type = "taskqueue"
    
    def __init__(self, broker_url: str, backend_url: str, **kwargs):
        self.broker_url = broker_url
        self.backend_url = backend_url
        self.app_name = kwargs.get('app_name', 'experiment_hub')
        self.app = None
    
    def initialize(self) -> None:
        """Initialize the plugin"""
        self.app = Celery(
            self.app_name,
            broker=self.broker_url,
            backend=self.backend_url
        )
        
        # Configure Celery
        self.app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
        )
    
    def shutdown(self) -> None:
        """Clean up resources"""
        # Nothing specific to clean up for Celery
        pass
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if Celery plugin is working correctly"""
        try:
            # Try to ping the broker
            ping_result = self.app.control.ping(timeout=1.0)
            if not ping_result:
                return False, {
                    "status": "error", 
                    "message": "No workers responded to ping", 
                    "broker_url": self.broker_url
                }
            
            # Check backend connection
            self.app.backend.client.ping()
            
            return True, {
                "status": "connected", 
                "broker_url": self.broker_url, 
                "backend_url": self.backend_url,
                "workers": len(ping_result)
            }
        except Exception as e:
            return False, {
                "status": "error", 
                "message": str(e), 
                "broker_url": self.broker_url, 
                "backend_url": self.backend_url
            }
    
    def register_tasks(self, tasks_module: str) -> None:
        """Register tasks from a module"""
        self.app.autodiscover_tasks([tasks_module])
    
    def submit_task(self, task_name: str, *args, **kwargs) -> str:
        """Submit a task to the queue"""
        async_result = self.app.send_task(task_name, args=args, kwargs=kwargs)
        return async_result.id
    
    def get_result(self, task_id: str, timeout: Optional[int] = None) -> Any:
        """Get the result of a task"""
        return self.app.AsyncResult(task_id).get(timeout=timeout)
    
    def get_status(self, task_id: str) -> str:
        """Get the status of a task"""
        return self.app.AsyncResult(task_id).status
