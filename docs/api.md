# ExperimentHub API

The ExperimentHub API provides a RESTful interface for interacting with the ExperimentHub. This document explains the available endpoints, their parameters, and example usage.

## API Overview

The ExperimentHub API is built using FastAPI and provides endpoints for:

1. **Health Checks**: Check the health of all plugins
2. **Experiment Management**: Create, load, save, and kill experiments
3. **Training and Prediction**: Train models and make predictions
4. **Visualization**: Get plots and figures from experiments

## Base URL

The API is typically available at:

```
http://localhost:8010
```

## Authentication

The API currently does not implement authentication. If you need to secure the API, consider using an API gateway or implementing authentication middleware.

## Endpoints

### Health and Status

#### GET /health

Check the health of all plugins.

**Response**:

```json
{
  "mlops": {
    "healthy": true,
    "details": {
      "status": "connected",
      "uri": "http://localhost:5000"
    }
  },
  "datastream": {
    "healthy": true,
    "details": {
      "status": "connected",
      "host": "localhost",
      "port": 5672
    }
  }
}
```

#### GET /available_experiments

Get available experiment types.

**Response**:

```json
{
  "time_series": {
    "module": "framework.TimeSeriesAnalysis",
    "class": "TimeSeriesExperiment"
  },
  "fault_detection": {
    "module": "framework.FaultDetection",
    "class": "FaultDetectionExperiment"
  }
}
```

### Experiment Management

#### POST /create_experiment/{name}/{experiment_type}

Create a new experiment of the specified type.

**Path Parameters**:

- `name`: Name of the experiment
- `experiment_type`: Type of experiment (must be registered)

**Request Body**:

Configuration for the experiment (JSON).

**Example**:

```bash
curl -X POST "http://localhost:8010/create_experiment/my_experiment/time_series" \
  -H "Content-Type: application/json" \
  -d '{
    "setup": {
      "target": "value",
      "ds": "ds"
    },
    "create_model": {
      "model_type": "Prophet",
      "params": {
        "seasonality_mode": "multiplicative",
        "daily_seasonality": true
      }
    }
  }'
```

**Response**:

```json
{
  "message": "Creating experiment 'my_experiment' of type 'time_series'"
}
```

#### POST /{name}/train

Train an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Request Body**:

Configuration for the experiment (JSON).

**Example**:

```bash
curl -X POST "http://localhost:8010/my_experiment/train" \
  -H "Content-Type: application/json" \
  -d '{
    "setup": {
      "target": "value",
      "ds": "ds"
    },
    "create_model": {
      "model_type": "Prophet",
      "params": {
        "seasonality_mode": "multiplicative",
        "daily_seasonality": true
      }
    }
  }'
```

**Response**:

```json
{
  "message": "Training started based on sent parameters."
}
```

#### POST /{name}/load

Load an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X POST "http://localhost:8010/my_experiment/load"
```

**Response**:

```
Experiment loaded
```

#### POST /{name}/load/{run_id}

Load an experiment with a specific run ID.

**Path Parameters**:

- `name`: Name of the experiment
- `run_id`: MLflow run ID

**Example**:

```bash
curl -X POST "http://localhost:8010/my_experiment/load/abcdef123456"
```

**Response**:

```
Experiment loaded
```

#### POST /{name}/save

Save an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X POST "http://localhost:8010/my_experiment/save"
```

**Response**:

```
Experiment saved
```

#### POST /{name}/kill

Kill an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X POST "http://localhost:8010/my_experiment/kill"
```

**Response**:

```
Experiment killed
```

### Training and Prediction

#### POST /{name}/predict

Make predictions with an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X POST "http://localhost:8010/my_experiment/predict"
```

**Response**:

```
Prediction called
```

#### POST /{name}/retrain

Retrain an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X POST "http://localhost:8010/my_experiment/retrain"
```

**Response**:

```
Retraining started as requested.
```

### Visualization

#### GET /{name}/plot/{plot_name}

Get a plot from an experiment.

**Path Parameters**:

- `name`: Name of the experiment
- `plot_name`: Name of the plot

**Example**:

```bash
curl -X GET "http://localhost:8010/my_experiment/plot/forecast"
```

**Response**:

Plotly JSON figure.

#### GET /{name}/plot_eda/{plot_name}

Get an EDA figure from an experiment.

**Path Parameters**:

- `name`: Name of the experiment
- `plot_name`: Name of the figure

**Example**:

```bash
curl -X GET "http://localhost:8010/my_experiment/plot_eda/correlation"
```

**Response**:

Plotly JSON figure.

### Data Management

#### POST /flush/{queue}

Flush a queue.

**Path Parameters**:

- `queue`: Name of the queue

**Example**:

```bash
curl -X POST "http://localhost:8010/flush/my_experiment"
```

**Response**:

```
Rabbit flushed
```

#### GET /{name}/train_data

Get training data from an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X GET "http://localhost:8010/my_experiment/train_data"
```

**Response**:

JSON data.

#### GET /{name}/cfg

Get configuration of an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X GET "http://localhost:8010/my_experiment/cfg"
```

**Response**:

JSON configuration.

### Metadata

#### GET /experiments

Get experiment names and figure names.

**Example**:

```bash
curl -X GET "http://localhost:8010/experiments"
```

**Response**:

```json
{
  "my_experiment": [
    ["forecast", "components"],
    ["correlation", "histogram"]
  ]
}
```

#### GET /{name}/run_id

Get run ID of an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X GET "http://localhost:8010/my_experiment/run_id"
```

**Response**:

```
abcdef123456
```

#### GET /{name}/exp_id

Get experiment ID of an experiment.

**Path Parameters**:

- `name`: Name of the experiment

**Example**:

```bash
curl -X GET "http://localhost:8010/my_experiment/exp_id"
```

**Response**:

```
123
```

## Error Handling

The API returns appropriate HTTP status codes for different error conditions:

- `200 OK`: The request was successful
- `204 No Content`: The request was successful, but there is no content to return
- `400 Bad Request`: The request was invalid
- `404 Not Found`: The requested resource was not found
- `500 Internal Server Error`: An error occurred on the server

Error responses include a detail message explaining the error:

```json
{
  "detail": "Unknown experiment type: invalid_type"
}
```

## API Client

Here's an example of a simple Python client for the ExperimentHub API:

```python
import requests
import json

class ExperimentHubClient:
    def __init__(self, base_url="http://localhost:8010"):
        self.base_url = base_url
    
    def check_health(self):
        """Check the health of all plugins"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_available_experiments(self):
        """Get available experiment types"""
        response = requests.get(f"{self.base_url}/available_experiments")
        return response.json()
    
    def create_experiment(self, name, experiment_type, config):
        """Create a new experiment"""
        response = requests.post(
            f"{self.base_url}/create_experiment/{name}/{experiment_type}",
            json=config
        )
        return response.json()
    
    def train(self, name, config):
        """Train an experiment"""
        response = requests.post(
            f"{self.base_url}/{name}/train",
            json=config
        )
        return response.text
    
    def predict(self, name):
        """Make predictions with an experiment"""
        response = requests.post(f"{self.base_url}/{name}/predict")
        return response.text
    
    def load(self, name, run_id=None):
        """Load an experiment"""
        if run_id:
            response = requests.post(f"{self.base_url}/{name}/load/{run_id}")
        else:
            response = requests.post(f"{self.base_url}/{name}/load")
        return response.text
    
    def save(self, name):
        """Save an experiment"""
        response = requests.post(f"{self.base_url}/{name}/save")
        return response.text
    
    def get_plot(self, name, plot_name):
        """Get a plot from an experiment"""
        response = requests.get(f"{self.base_url}/{name}/plot/{plot_name}")
        return response.json()
    
    def get_eda_plot(self, name, plot_name):
        """Get an EDA figure from an experiment"""
        response = requests.get(f"{self.base_url}/{name}/plot_eda/{plot_name}")
        return response.json()
    
    def get_experiments(self):
        """Get experiment names and figure names"""
        response = requests.get(f"{self.base_url}/experiments")
        return response.json()
```

## Example Usage

```python
# Create client
client = ExperimentHubClient()

# Check health
health = client.check_health()
print(f"Health: {json.dumps(health, indent=2)}")

# Get available experiments
experiments = client.get_available_experiments()
print(f"Available experiments: {json.dumps(experiments, indent=2)}")

# Create and train experiment
config = {
    "setup": {
        "target": "value",
        "ds": "ds"
    },
    "create_model": {
        "model_type": "Prophet",
        "params": {
            "seasonality_mode": "multiplicative",
            "daily_seasonality": True
        }
    }
}
result = client.create_experiment("my_experiment", "time_series", config)
print(f"Create result: {result}")

# Make predictions
prediction = client.predict("my_experiment")
print(f"Prediction result: {prediction}")

# Get plots
plot = client.get_plot("my_experiment", "forecast")
print(f"Plot: {plot}")
```

## Troubleshooting

### Connection Refused

If you get a "Connection refused" error, make sure the API server is running and accessible at the specified URL.

### Experiment Not Found

If you get a "Experiment not found" error, make sure the experiment exists and is loaded.

### Plugin Health Issues

If the health check shows that a plugin is unhealthy, check the plugin configuration and make sure the required services (MLflow, RabbitMQ, etc.) are running.
