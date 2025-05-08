import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import numpy as np
import pandas as pd
import asyncio
from framework.ExperimentHub import ExperimentHub

async def main():
    """
    Manual test script for the ExperimentHub plugin system.
    
    This script:
    1. Initializes the ExperimentHub from configuration
    2. Checks the health of all plugins
    3. Creates and trains an experiment
    4. Makes predictions with the trained experiment
    
    Prerequisites:
    - MLflow server running on http://localhost:5000
    - RabbitMQ server running on localhost:5672
    - Celery workers running (optional)
    """
    print("Starting manual test of ExperimentHub plugin system...")
    
    # Initialize ExperimentHub from configuration
    print("Initializing ExperimentHub from configuration...")
    hub = ExperimentHub.from_config("hub_config.yaml")
    
    # Check plugin health
    print("\nChecking plugin health...")
    health_info = hub.check_plugin_health()
    for plugin_type, info in health_info.items():
        status = "HEALTHY" if info["healthy"] else "UNHEALTHY"
        print(f"  {plugin_type}: {status}")
        print(f"    Details: {info['details']}")
    
    # List available experiment types
    print("\nAvailable experiment types:")
    for exp_type, exp_info in hub.available_experiments.items():
        print(f"  {exp_type}: {exp_info['module']}.{exp_info['class']}")
    
    # Create and train an experiment
    experiment_type = "time_series"
    experiment_name = "test_experiment"
    
    # Example configuration for a time series experiment
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
        },
        # Add load_object information to config
        "load_object": {
            "module": "framework.TimeSeriesAnalysis",
            "name": "TimeSeriesExperiment"
        }
    }
    
    # Get data stream plugin
    datastream_plugin = hub.get_plugin("datastream")
    if not datastream_plugin:
        print("\nERROR: DataStream plugin not found. Cannot continue with training.")
        return
    
    # Create queue and send data
    print(f"\nCreating queue '{experiment_name}'...")
    datastream_plugin.create_queue(experiment_name)
    
    # Example data
    print("Generating example data...")
    data = pd.DataFrame({
        "ds": pd.date_range(start="2020-01-01", periods=100),
        "value": np.random.normal(0, 1, 100)
    })
    
    # Send data to queue
    print(f"Sending data to queue '{experiment_name}'...")
    datastream_plugin.pull_data = lambda queue: data.to_json()
    
    # Train experiment
    print(f"\nTraining experiment '{experiment_name}'...")
    try:
        result = await hub.train(experiment_name, config)
        print(f"Training result: {result}")
    except Exception as e:
        print(f"ERROR during training: {e}")
        return
    
    # Make predictions
    print(f"\nMaking predictions with experiment '{experiment_name}'...")
    try:
        predictions = await hub.predict(experiment_name)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Prediction sample: {predictions.head()}")
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return
    
    print("\nManual test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
