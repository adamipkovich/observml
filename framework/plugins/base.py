from typing import Protocol, Any, Tuple, Dict

class Plugin(Protocol):
    """Base interface for all plugins"""
    plugin_type: str
    
    def __init__(self, **kwargs) -> None:
        """Initialize the plugin with configuration"""
        ...
    
    def shutdown(self) -> None:
        """Clean up resources when shutting down"""
        ...
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if the plugin is working correctly
        
        Returns:
            Tuple containing:
            - Boolean indicating if the plugin is healthy
            - Dictionary with additional health information and metrics
        """
        ...
