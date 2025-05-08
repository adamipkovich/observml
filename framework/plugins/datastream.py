from framework.plugins.base import Plugin
from typing import Any

class DataStreamPlugin(Plugin):
    """Plugin for data streaming operations"""
    plugin_type = "datastream"
    
    def connect(self, **kwargs) -> None:
        """Connect to the data stream"""
        ...
    
    def disconnect(self) -> None:
        """Disconnect from the data stream"""
        ...
    
    def create_queue(self, queue_name: str) -> None:
        """Create a new queue"""
        ...
    
    def pull_data(self, queue: str) -> Any:
        """Pull data from the queue"""
        ...
    
    def flush_queue(self, queue: str) -> None:
        """Flush the queue"""
        ...
