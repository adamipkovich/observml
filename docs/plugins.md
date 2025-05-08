# ExperimentHub Plugin System

The ExperimentHub plugin system provides a flexible and extensible architecture for integrating various services and functionalities into the ExperimentHub. This document explains the plugin architecture, how to use existing plugins, and how to create custom plugins.

## Plugin Architecture

The plugin system is based on a Protocol-based architecture, where each plugin type defines an interface that implementations must adhere to. The main components of the plugin system are:

```mermaid
graph TD
    A[ExperimentHub] --> B[Plugin System]
    B --> C[MLOps Plugin]
    B --> D[DataStream Plugin]
    B --> E[TaskQueue Plugin]
    C --> F[MLflowPlugin]
    D --> G[RabbitMQPlugin]
    E --> H[CeleryPlugin]
```

### Plugin Types

The ExperimentHub supports three main types of plugins:

1. **MLOps Plugin**: Handles model tracking, versioning, and artifact storage
   - Default implementation: `MLflowPlugin`

2. **DataStream Plugin**: Handles data streaming and messaging
   - Default implementation: `RabbitMQPlugin`

3. **TaskQueue Plugin**: Handles asynchronous task processing
   - Default implementation: `CeleryPlugin`

### Plugin Interface

All plugins must implement the `Plugin` Protocol defined in `framework/plugins/base.py`:

```python
class Plugin(Protocol):
    """Base interface for all plugins"""
    plugin_type: str
    
    def __init__(self, **kwargs) -> None:
        """Initialize the plugin with configuration"""
        ...
    
    def initialize(self) -> None:
        """Initialize the plugin"""
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
```

Each plugin type then extends this base interface with additional methods specific to its functionality.

## Configuration System

The ExperimentHub plugin system is configured using a YAML configuration file (`hub_config.yaml`). This file specifies which plugins to use, their configurations, and available experiment types.

### Configuration File Structure

```yaml
# hub_config.yaml
version: "1.0"

# Global settings
global:
  log_level: "info"
  environment: "development"

# Plugin configurations
plugins:
  # MLOps plugin configuration
  mlops:
    enabled: true
    type: "mlflow"  # Which implementation to use
    config:
      mlflow_uri: "http://localhost:5000"
  
  # Data stream plugin configuration
  datastream:
    enabled: true
    type: "rabbitmq"
    config:
      host: "localhost"
      port: 5672
      username: "guest"
      password: "guest"
  
  # Task queue plugin configuration
  taskqueue:
    enabled: false  # Disabled by default, enable when needed
    type: "celery"
    config:
      broker_url: "amqp://guest:guest@localhost:5672//"
      backend_url: "redis://localhost:6379/0"
      app_name: "experiment_hub"

# Available experiment types
experiments:
  - name: "time_series"
    module: "framework.TimeSeriesAnalysis"
    class: "TimeSeriesExperiment"
    enabled: true
  
  - name: "fault_detection"
    module: "framework.FaultDetection"
    class: "FaultDetectionExperiment"
    enabled: true
```

### Loading Configuration

The ExperimentHub can be initialized from a configuration file using the `from_config` class method:

```python
from framework.ExperimentHub import ExperimentHub

# Initialize from configuration file
hub = ExperimentHub.from_config("hub_config.yaml")
```

## Using Plugins

Once the ExperimentHub is initialized with plugins, you can use them through the ExperimentHub API:

```python
# Get a plugin by type
mlops_plugin = hub.get_plugin("mlops")
datastream_plugin = hub.get_plugin("datastream")
taskqueue_plugin = hub.get_plugin("taskqueue")

# Check plugin health
health_info = hub.check_plugin_health()
```

## Creating Custom Plugins

You can create custom plugins by implementing the appropriate plugin interface. Here's an example of creating a custom MLOps plugin:

```python
from framework.plugins.mlops import MLOpsPlugin
from typing import Any, Dict, Tuple

class CustomMLOpsPlugin(MLOpsPlugin):
    """Custom implementation of MLOpsPlugin"""
    plugin_type = "mlops"
    
    def __init__(self, custom_param: str, **kwargs):
        self.custom_param = custom_param
    
    def initialize(self) -> None:
        """Initialize the plugin"""
        # Custom initialization logic
        pass
    
    def shutdown(self) -> None:
        """Clean up resources"""
        # Custom cleanup logic
        pass
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if plugin is working correctly"""
        try:
            # Custom health check logic
            return True, {"status": "connected", "custom_param": self.custom_param}
        except Exception as e:
            return False, {"status": "error", "message": str(e)}
    
    # Implement other MLOpsPlugin methods
    # ...
```

To use your custom plugin, you can register it with the ExperimentHub:

```python
from framework.ExperimentHub import ExperimentHub

# Create ExperimentHub
hub = ExperimentHub()

# Create and register custom plugin
custom_plugin = CustomMLOpsPlugin(custom_param="value")
hub.register_plugin(custom_plugin)
```

## Plugin Health Checks

All plugins must implement a `health_check` method that verifies if the plugin is working correctly. This method returns a tuple containing:

1. A boolean indicating if the plugin is healthy
2. A dictionary with additional health information and metrics

You can check the health of all registered plugins using the `check_plugin_health` method of the ExperimentHub:

```python
health_info = hub.check_plugin_health()
for plugin_type, info in health_info.items():
    status = "HEALTHY" if info["healthy"] else "UNHEALTHY"
    print(f"{plugin_type}: {status}")
    print(f"  Details: {info['details']}")
```

## Plugin Lifecycle

Plugins go through the following lifecycle:

1. **Initialization**: The plugin is created with configuration parameters
2. **Registration**: The plugin is registered with the ExperimentHub
3. **Initialization**: The `initialize` method is called to set up the plugin
4. **Usage**: The plugin is used by the ExperimentHub
5. **Shutdown**: The `shutdown` method is called to clean up resources

## Best Practices

When working with plugins, follow these best practices:

1. **Error Handling**: Plugins should handle errors gracefully and provide meaningful error messages
2. **Resource Management**: Plugins should clean up resources in the `shutdown` method
3. **Configuration**: Plugins should accept configuration parameters in the constructor
4. **Health Checks**: Plugins should implement thorough health checks
5. **Documentation**: Document the purpose, configuration, and usage of your plugins
