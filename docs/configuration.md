# ExperimentHub Configuration System

The ExperimentHub configuration system provides a flexible way to configure the ExperimentHub, its plugins, and available experiment types. This document explains the configuration file structure, how to load configuration, and best practices for configuration management.

## Configuration File Structure

The ExperimentHub is configured using a YAML configuration file (`hub_config.yaml`). The file has the following structure:

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
  
  - name: "fault_isolation"
    module: "framework.FaultIsolation"
    class: "FaultIsolationExperiment"
    enabled: true
  
  - name: "process_mining"
    module: "framework.ProcessMining"
    class: "ProcessMiningExperiment"
    enabled: true
```

### Global Settings

The `global` section contains global settings for the ExperimentHub:

- `log_level`: The logging level (e.g., "debug", "info", "warning", "error")
- `environment`: The environment (e.g., "development", "testing", "production")

### Plugin Configurations

The `plugins` section contains configurations for each plugin type:

- `mlops`: Configuration for the MLOps plugin
- `datastream`: Configuration for the DataStream plugin
- `taskqueue`: Configuration for the TaskQueue plugin

Each plugin configuration has the following structure:

- `enabled`: Whether the plugin is enabled
- `type`: The type of plugin implementation to use
- `config`: Configuration parameters specific to the plugin implementation

### Experiment Types

The `experiments` section contains a list of available experiment types:

- `name`: The name of the experiment type
- `module`: The Python module containing the experiment class
- `class`: The name of the experiment class
- `enabled`: Whether the experiment type is enabled

## Loading Configuration

The ExperimentHub can be initialized from a configuration file using the `from_config` class method:

```python
from framework.ExperimentHub import ExperimentHub

# Initialize from configuration file
hub = ExperimentHub.from_config("hub_config.yaml")
```

The `from_config` method performs the following steps:

1. Loads the configuration file
2. Creates an ExperimentHub instance
3. Initializes plugins based on the configuration
4. Registers available experiment types
5. Returns the initialized ExperimentHub

## Environment Variables

The ExperimentHubAPI also supports configuration through environment variables:

- `HUB_CONFIG_PATH`: Path to the configuration file (default: "hub_config.yaml")
- `MLFLOW_URI`: URI for MLflow tracking server
- `RABBIT_HOST`: Hostname for RabbitMQ server
- `RABBIT_PORT`: Port for RabbitMQ server
- `RABBIT_USER`: Username for RabbitMQ server
- `RABBIT_PASSWORD`: Password for RabbitMQ server

If the configuration file specified by `HUB_CONFIG_PATH` exists, it will be used. Otherwise, the ExperimentHub will be initialized using environment variables.

## Configuration Best Practices

### Separate Configuration by Environment

Create separate configuration files for different environments:

- `hub_config.dev.yaml`: Development environment
- `hub_config.test.yaml`: Testing environment
- `hub_config.prod.yaml`: Production environment

### Use Environment Variables for Sensitive Information

Use environment variables for sensitive information like passwords and API keys:

```yaml
plugins:
  datastream:
    enabled: true
    type: "rabbitmq"
    config:
      host: "localhost"
      port: 5672
      username: "${RABBIT_USER}"
      password: "${RABBIT_PASSWORD}"
```

### Document Configuration Parameters

Document all configuration parameters, including their purpose, allowed values, and default values.

### Validate Configuration

Validate the configuration file before using it:

```python
def validate_config(config):
    """Validate the configuration file"""
    # Check required sections
    required_sections = ["version", "plugins", "experiments"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Check plugin configurations
    for plugin_type, plugin_config in config["plugins"].items():
        if "enabled" not in plugin_config:
            raise ValueError(f"Missing 'enabled' field in {plugin_type} plugin configuration")
        if plugin_config["enabled"] and "type" not in plugin_config:
            raise ValueError(f"Missing 'type' field in {plugin_type} plugin configuration")
        if plugin_config["enabled"] and "config" not in plugin_config:
            raise ValueError(f"Missing 'config' field in {plugin_type} plugin configuration")
    
    # Check experiment configurations
    for i, exp_config in enumerate(config["experiments"]):
        if "name" not in exp_config:
            raise ValueError(f"Missing 'name' field in experiment configuration at index {i}")
        if "module" not in exp_config:
            raise ValueError(f"Missing 'module' field in experiment configuration at index {i}")
        if "class" not in exp_config:
            raise ValueError(f"Missing 'class' field in experiment configuration at index {i}")
```

### Use Default Values

Provide default values for optional configuration parameters:

```python
def get_config_value(config, path, default=None):
    """Get a value from the configuration, with a default value"""
    parts = path.split(".")
    current = config
    for part in parts:
        if part not in current:
            return default
        current = current[part]
    return current
```

## Example Configurations

### Development Configuration

```yaml
# hub_config.dev.yaml
version: "1.0"

global:
  log_level: "debug"
  environment: "development"

plugins:
  mlops:
    enabled: true
    type: "mlflow"
    config:
      mlflow_uri: "http://localhost:5000"
  
  datastream:
    enabled: true
    type: "rabbitmq"
    config:
      host: "localhost"
      port: 5672
      username: "guest"
      password: "guest"
  
  taskqueue:
    enabled: false
    type: "celery"
    config:
      broker_url: "amqp://guest:guest@localhost:5672//"
      backend_url: "redis://localhost:6379/0"
      app_name: "experiment_hub"

experiments:
  - name: "time_series"
    module: "framework.TimeSeriesAnalysis"
    class: "TimeSeriesExperiment"
    enabled: true
```

### Production Configuration

```yaml
# hub_config.prod.yaml
version: "1.0"

global:
  log_level: "info"
  environment: "production"

plugins:
  mlops:
    enabled: true
    type: "mlflow"
    config:
      mlflow_uri: "http://mlflow.example.com"
  
  datastream:
    enabled: true
    type: "rabbitmq"
    config:
      host: "rabbitmq.example.com"
      port: 5672
      username: "${RABBIT_USER}"
      password: "${RABBIT_PASSWORD}"
  
  taskqueue:
    enabled: true
    type: "celery"
    config:
      broker_url: "amqp://${RABBIT_USER}:${RABBIT_PASSWORD}@rabbitmq.example.com:5672//"
      backend_url: "redis://redis.example.com:6379/0"
      app_name: "experiment_hub"

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

## Troubleshooting

### Configuration File Not Found

If the configuration file is not found, the ExperimentHub will fall back to using environment variables. Make sure the configuration file exists and is accessible.

### Invalid Configuration

If the configuration file is invalid, the ExperimentHub will raise an error. Check the error message for details on what's wrong with the configuration.

### Plugin Initialization Errors

If a plugin fails to initialize, check the plugin configuration and make sure all required parameters are provided.
