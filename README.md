# ObservML

**Modular Machine Learning for Anomaly Detection and Process Monitoring**

ObservML is a comprehensive machine learning framework designed for process monitoring, anomaly detection, fault isolation, and time series analysis. Built with a plugin-based architecture, it provides easy-to-use microservices with integrated MLOps, tracking, and deployment capabilities.

## Why ObservML?

ObservML was created to provide a robust platform for monitoring industrial processes while maintaining access to state-of-the-art AI and infrastructure tools. The framework offers:

- **Process Mining, Fault Detection, Fault Isolation, and Time Series Analysis** tools specifically designed for predictive maintenance
- **Integrated MLOps** with MLflow for experiment tracking and model registry
- **Plugin Architecture** for extensible functionality and easy integration
- **Configuration-Based Model Behavior** for simplified interaction and deployment
- **FastAPI-based REST API** for easy integration with existing systems
- **RabbitMQ Message Brokering** for real-time data streaming and predictions
- **Docker-Ready Deployment** without the complexity and costs of Kubernetes

## Architecture

ObservML is built around the **ExperimentHub**, a central component that manages multiple experiments through a plugin system:

- **MLOps Plugin**: Handles model versioning and experiment tracking (MLflow)
- **DataStream Plugin**: Manages real-time data streaming (RabbitMQ)
- **Experiment Types**: Modular experiment implementations for different use cases

## Features

### 🔧 Core Components

- **ExperimentHub**: Central management system for all experiments
- **Plugin System**: Extensible architecture for adding new functionality
- **Configuration Management**: YAML-based configuration for easy setup
- **REST API**: FastAPI-based API for programmatic access

### 🤖 Machine Learning Capabilities

- **Time Series Analysis**: ARIMA, Prophet, LSTM, Autoencoder, SSA
- **Fault Detection**: Isolation Forest, DBSCAN, Elliptic Envelope, PCA
- **Fault Isolation**: Decision Trees, Naive Bayes, HMM, Bayesian Networks
- **Process Mining**: Apriori, CMSPAM, TopK Rules, Heuristics Miner

### 🚀 Infrastructure

- **MLflow Integration**: Experiment tracking, model registry, and versioning
- **RabbitMQ**: Message brokering for continuous prediction and data streaming
- **Docker Support**: Containerized deployment with docker-compose
- **FastAPI**: High-performance REST API with automatic documentation

### 📊 Visualization

- **Plotly Integration**: Interactive plots and dashboards
- **EDA Tools**: Exploratory data analysis with built-in visualizations
- **Real-time Monitoring**: Live prediction results and model performance

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- 16GB+ RAM recommended
- 32-64GB free disk space

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adamipkovich/observml.git
   cd observml
   ```

2. **Install dependencies:**
   ```bash
   pip install poetry
   poetry install
   ```

3. **Start infrastructure services:**
   ```bash
   docker-compose up -d mlflow rabbitmq
   ```

4. **Configure the system:**
   ```bash
   cp hub_config.yaml.example hub_config.yaml
   # Edit hub_config.yaml with your settings
   ```

5. **Start the API server:**
   ```bash
   python ExperimentHubAPI.py
   ```

The API will be available at `http://localhost:8010` with interactive documentation at `http://localhost:8010/docs`.

### Docker Deployment

For a complete containerized deployment:

```bash
docker-compose up -d
```

This starts:
- MLflow tracking server (port 5000)
- RabbitMQ message broker (port 5672)
- ObservML API server (port 8010)
- Streamlit frontend (port 8501)

## Usage Example

```python
from framework.ExperimentHub import ExperimentHub

# Initialize from configuration
hub = ExperimentHub.from_config("hub_config.yaml")

# Create a time series experiment
config = {
    "setup": {
        "target": "value",
        "ds": "timestamp"
    },
    "create_model": {
        "model_type": "Prophet",
        "params": {
            "seasonality_mode": "multiplicative",
            "daily_seasonality": True
        }
    }
}

# Train the experiment
await hub.train("my_experiment", config)

# Make predictions
predictions = await hub.predict("my_experiment")

# Get visualizations
plot = hub.plot("my_experiment", "forecast")
```

## API Usage

The REST API provides endpoints for all major operations:

```bash
# Check system health
curl http://localhost:8010/health

# Create and train an experiment
curl -X POST "http://localhost:8010/create_experiment/my_exp/time_series" \
  -H "Content-Type: application/json" \
  -d '{"setup": {"target": "value"}, "create_model": {"model_type": "Prophet"}}'

# Make predictions
curl -X POST "http://localhost:8010/my_exp/predict"

# Get plots
curl "http://localhost:8010/my_exp/plot/forecast"
```

## Configuration

ObservML uses YAML configuration files for setup. Example `hub_config.yaml`:

```yaml
version: "1.0"

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

experiments:
  - name: "time_series"
    module: "framework.TimeSeriesAnalysis"
    class: "TimeSeriesExperiment"
    enabled: true
```

## Documentation

- **[Getting Started](docs/index.md)**: Detailed setup and first steps
- **[API Reference](docs/api.md)**: Complete API documentation
- **[Configuration Guide](docs/configuration.md)**: Configuration options and examples
- **[Plugin System](docs/plugin_system.md)**: Extending ObservML with plugins
- **[Deployment Guide](docs/serve.md)**: Production deployment instructions

## Development

### Project Structure

```
observml/
├── framework/           # Core framework components
│   ├── ExperimentHub.py # Central experiment management
│   ├── Experiment.py    # Base experiment class
│   └── plugins/         # Plugin implementations
├── models/              # ML model implementations
├── configs/             # Configuration templates
├── data/               # Sample datasets
├── docs/               # Documentation
└── tests/              # Test suite
```

### Adding New Experiment Types

1. Create a new experiment class inheriting from `Experiment`
2. Implement required methods (`train`, `predict`, `retrain`)
3. Add configuration to `hub_config.yaml`
4. Register in the experiments list

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Roadmap

### Current Development
- Enhanced testing suite (data, experiment, API, microservices tests)
- Improved error handling and logging
- Performance optimizations

### Planned Features
- **LLM Integration**: Dependency management and Docker image building
- **Advanced Visualization**: Interactive dashboards and monitoring frontend
- **Extended Model Library**: More neural networks, time series models, and SPMF algorithms
- **Feature Engineering**: Automated feature selection and engineering
- **Model Interpretability**: SHAP, LIME, and other explainability tools
- **Advanced Analytics**: Sensitivity analysis, impulse-response, optimization solutions

### Infrastructure Improvements
- **Load Balancing**: NGINX integration
- **CI/CD Pipeline**: Automated testing and deployment
- **Data Pipeline**: DBT for transformations, Meltano for ETL
- **Orchestration**: Apache Airflow support
- **Notebook Integration**: Jupyter notebook support for pipelines

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use ObservML in your research, please cite:

```bibtex
@software{observml2024,
  title={ObservML: Modular Machine Learning for Anomaly Detection},
  author={ObservML Team},
  year={2024},
  url={https://github.com/adamipkovich/observml}
}
```

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/adamipkovich/observml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/adamipkovich/observml/discussions)

## Acknowledgments

This project was developed as part of research project 2020-1.1.2-PIACI-KFI-2020-00062, focusing on modular machine learning workflows for industrial anomaly detection.
