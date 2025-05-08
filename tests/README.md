# ExperimentHub Plugin System Tests

This directory contains tests for the ExperimentHub plugin system. The tests are organized into several categories:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test how components work together
3. **Manual Tests**: Scripts for testing with real services

## Prerequisites

Before running the tests, make sure you have the following dependencies installed:

```bash
pip install pytest pytest-mock fastapi requests pandas numpy
```

For manual tests, you'll also need:

1. MLflow server running on http://localhost:5000
2. RabbitMQ server running on localhost:5672
3. Celery workers running (optional)

## Running Unit Tests

To run all unit tests:

```bash
pytest tests/test_plugins.py tests/test_experiment_hub.py tests/test_api.py -v
```

To run a specific test file:

```bash
pytest tests/test_plugins.py -v
```

## Running Manual Tests

### Testing the ExperimentHub directly

The `manual_test.py` script tests the ExperimentHub directly, without going through the API:

```bash
python tests/manual_test.py
```

This script:
1. Initializes the ExperimentHub from configuration
2. Checks the health of all plugins
3. Creates and trains an experiment
4. Makes predictions with the trained experiment

### Testing the API endpoints

The `test_api_endpoints.py` script tests the API endpoints:

```bash
python tests/test_api_endpoints.py --base-url http://localhost:8010
```

This script:
1. Tests the health endpoint
2. Tests the available experiments endpoint
3. Tests creating an experiment
4. Tests making predictions

## Test Files

- `test_plugins.py`: Unit tests for plugin implementations
- `test_experiment_hub.py`: Unit tests for the ExperimentHub class
- `test_api.py`: Unit tests for the API endpoints
- `manual_test.py`: Manual test script for the ExperimentHub
- `test_api_endpoints.py`: Manual test script for the API endpoints

## Adding New Tests

When adding new tests, follow these guidelines:

1. **Unit Tests**: Test a single component in isolation, using mocks for dependencies
2. **Integration Tests**: Test how components work together, using mocks for external services
3. **Manual Tests**: Test with real services, providing clear instructions for setup

## Troubleshooting

If you encounter issues with the tests:

1. **Import Errors**: Make sure the project root is in your Python path
2. **Connection Errors**: Check that required services (MLflow, RabbitMQ) are running
3. **Mock Errors**: Check that mocks are configured correctly
