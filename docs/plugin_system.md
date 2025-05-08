# ExperimentHub Plugin System

The ExperimentHub plugin system provides a flexible and extensible architecture for integrating various services and functionalities into the ExperimentHub. This document provides a comprehensive overview of the plugin system, including its architecture, configuration, and usage.

## System Architecture

```mermaid
graph TD
    A[ExperimentHub] --> B[Plugin System]
    B --> C[MLOps Plugin]
    B --> D[DataStream Plugin]
    B --> E[TaskQueue Plugin]
    A --> F[Experiment Management]
    F --> G[Training]
    F --> H[Prediction]
    F --> I[Retraining]
    A --> J[Configuration System]
    J --> K[Plugin Configuration]
    J --> L[Experiment Configuration]
    C --> M[MLflowPlugin]
    D --> N[RabbitMQPlugin]
    E --> O[CeleryPlugin]
```

The ExperimentHub plugin system consists of three main components:

1. **Plugin Interfaces**: Define the contract that plugin implementations must adhere to
2. **Plugin Implementations**: Concrete implementations of the plugin interfaces
3. **Plugin Registry**: Manages plugin registration and retrieval

## Plugin Types

The ExperimentHub supports three main types of plugins:

### 1. MLOps Plugin

The MLOps plugin handles model tracking, versioning, and artifact storage. It provides methods for:

- Creating experiments
- Starting runs
- Logging artifacts
- Retrieving runs and experiments

Default implementation: `MLflowPlugin`

### 2. DataStream Plugin

The DataStream plugin handles data streaming and messaging. It provides methods for:

- Creating queues
- Sending and receiving data
- Flushing queues

Default implementation: `RabbitMQPlugin`

### 3. TaskQueue Plugin

The TaskQueue plugin handles asynchronous task processing. It provides methods for:

- Submitting tasks
- Getting task results
- Checking task status

Default implementation: `CeleryPlugin`

## Plugin Interfaces

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

### MLOps Plugin Interface

```python
class MLOpsPlugin(Plugin, Protocol):
    """Interface for MLOps plugins"""
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
```

### DataStream Plugin Interface

```python
class DataStreamPlugin(Plugin, Protocol):
    """Interface for data stream plugins"""
    plugin_type = "datastream"
    
    def connect(self, **kwargs) -> None:
        """Connect to the data stream service"""
        ...
    
    def disconnect(self) -> None:
        """Disconnect from the data stream service"""
        ...
    
    def create_queue(self, queue_name: str) -> None:
        """Create a queue"""
        ...
    
    def pull_data(self, queue: str) -> Any:
        """Pull data from a queue"""
        ...
    
    def flush_queue(self, queue: str) -> None:
        """Flush a queue"""
        ...
```

### TaskQueue Plugin Interface

```python
class TaskQueuePlugin(Plugin, Protocol):
    """Interface for task queue plugins"""
    plugin_type = "taskqueue"
    
    def register_tasks(self, tasks_module: str) -> None:
        """Register tasks from a module"""
        ...
    
    def submit_task(self, task_name: str, *args, **kwargs) -> str:
        """Submit a task to the queue"""
        ...
    
    def get_result(self, task_id: str, timeout: Optional[int] = None) -> Any:
        """Get the result of a task"""
        ...
    
    def get_status(self, task_id: str) -> str:
        """Get the status of a task"""
        ...
```

## Plugin Implementations

### MLflowPlugin

The `MLflowPlugin` is an implementation of the `MLOpsPlugin` interface that uses MLflow for model tracking and versioning.

```python
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
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if MLflow plugin is working correctly"""
        try:
            # Try to connect to MLflow server
            self.client.list_experiments()
            return True, {"status": "connected", "uri": self.mlflow_uri}
        except Exception as e:
            return False, {"status": "error", "message": str(e), "uri": self.mlflow_uri}
    
    # ... other methods ...
```

### RabbitMQPlugin

The `RabbitMQPlugin` is an implementation of the `DataStreamPlugin` interface that uses RabbitMQ for data streaming and messaging.

```python
class RabbitMQPlugin(DataStreamPlugin):
    """RabbitMQ implementation of DataStreamPlugin"""
    plugin_type = "datastream"
    
    def __init__(self, host: str, port: str|int, username: str, password: str, **kwargs):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None
        self.channel = None
    
    def initialize(self) -> None:
        """Initialize the plugin"""
        self.connect()
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if RabbitMQ plugin is working correctly"""
        try:
            # Check if connection is open
            if self.connection is None or self.connection.is_closed:
                self.connect()
            
            # Try to create a test queue
            test_queue = f"health_check_{int(time.time())}"
            self.channel.queue_declare(queue=test_queue, durable=False)
            self.channel.queue_delete(queue=test_queue)
            
            return True, {
                "status": "connected", 
                "host": self.host, 
                "port": self.port
            }
        except Exception as e:
            return False, {
                "status": "error", 
                "message": str(e), 
                "host": self.host, 
                "port": self.port
            }
    
    # ... other methods ...
```

### CeleryPlugin

The `CeleryPlugin` is an implementation of the `TaskQueuePlugin` interface that uses Celery for asynchronous task processing.

```python
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
    
    # ... other methods ...
```

## Plugin Registration and Retrieval

Plugins are registered with the ExperimentHub using the `register_plugin` method:

```python
def register_plugin(self, plugin: Plugin) -> None:
    """Register a plugin.
    
    Args:
        plugin: The plugin to register.
    """
    self.plugins[plugin.plugin_type] = plugin
    plugin.initialize()
```

Plugins can be retrieved using the `get_plugin` method:

```python
def get_plugin(self, plugin_type: str) -> Optional[Plugin]:
    """Get a plugin by type.
    
    Args:
        plugin_type: The type of plugin to get.
    
    Returns:
        The plugin, or None if not found.
    """
    return self.plugins.get(plugin_type)
```

## Plugin Health Checks

All plugins must implement a `health_check` method that verifies if the plugin is working correctly. This method returns a tuple containing:

1. A boolean indicating if the plugin is healthy
2. A dictionary with additional health information and metrics

The ExperimentHub provides a `check_plugin_health` method that checks the health of all registered plugins:

```python
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
```

## Plugin Configuration

Plugins are configured using the `hub_config.yaml` file:

```yaml
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
```

The ExperimentHub initializes plugins based on this configuration:

```python
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

## Plugin Lifecycle

Plugins go through the following lifecycle:

1. **Initialization**: The plugin is created with configuration parameters
2. **Registration**: The plugin is registered with the ExperimentHub
3. **Initialization**: The `initialize` method is called to set up the plugin
4. **Usage**: The plugin is used by the ExperimentHub
5. **Shutdown**: The `shutdown` method is called to clean up resources

## Best Practices

When working with plugins, follow these best practices:

### Error Handling

Plugins should handle errors gracefully and provide meaningful error messages:

```python
def health_check(self) -> Tuple[bool, Dict[str, Any]]:
    """Check if plugin is working correctly"""
    try:
        # Health check logic
        return True, {"status": "connected"}
    except ConnectionError as e:
        return False, {"status": "error", "message": f"Connection error: {str(e)}"}
    except TimeoutError as e:
        return False, {"status": "error", "message": f"Timeout error: {str(e)}"}
    except Exception as e:
        return False, {"status": "error", "message": f"Unexpected error: {str(e)}"}
```

### Resource Management

Plugins should clean up resources in the `shutdown` method:

```python
def shutdown(self) -> None:
    """Clean up resources"""
    if self.connection is not None:
        try:
            self.connection.close()
        except Exception as e:
            logging.error(f"Error closing connection: {e}")
        finally:
            self.connection = None
```

### Configuration

Plugins should accept configuration parameters in the constructor:

```python
def __init__(self, host: str, port: int, username: str, password: str, **kwargs):
    self.host = host
    self.port = port
    self.username = username
    self.password = password
    self.timeout = kwargs.get("timeout", 30)
    self.connection = None
```

### Health Checks

Plugins should implement thorough health checks:

```python
def health_check(self) -> Tuple[bool, Dict[str, Any]]:
    """Check if plugin is working correctly"""
    try:
        # Check connection
        if self.connection is None or not self.connection.is_connected():
            self.connect()
        
        # Check functionality
        self.ping()
        
        # Return health information
        return True, {
            "status": "connected",
            "host": self.host,
            "port": self.port,
            "connection_id": self.connection.id
        }
    except Exception as e:
        return False, {
            "status": "error",
            "message": str(e),
            "host": self.host,
            "port": self.port
        }
```

### Documentation

Document the purpose, configuration, and usage of your plugins:

```python
class CustomPlugin(Plugin):
    """Custom plugin for integrating with Example Service.
    
    This plugin provides integration with Example Service, allowing
    the ExperimentHub to use Example Service for specific functionality.
    
    Configuration:
        api_key: API key for authenticating with Example Service
        host: Hostname of the Example Service API
        port: Port of the Example Service API
        timeout: Timeout for API requests (default: 30 seconds)
    
    Usage:
        plugin = CustomPlugin(
            api_key="your-api-key",
            host="api.example.com",
            port=443,
            timeout=60
        )
        hub.register_plugin(plugin)
    """
    # ...
```

## Troubleshooting

### Plugin Initialization Errors

If a plugin fails to initialize, check the plugin configuration and make sure all required parameters are provided.

### Plugin Health Check Failures

If a plugin health check fails, check the health information for details on what went wrong.

### Missing Plugins

If a plugin is missing, check the configuration file and make sure the plugin is enabled and configured correctly.
