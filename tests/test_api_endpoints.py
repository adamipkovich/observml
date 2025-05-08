import requests
import json
import time
import argparse

def main(base_url="http://localhost:8010"):
    """
    Test script for the ExperimentHub API endpoints.
    
    This script:
    1. Tests the health endpoint
    2. Tests the available experiments endpoint
    3. Tests creating an experiment
    4. Tests making predictions
    
    Prerequisites:
    - ExperimentHubAPI running on the specified base_url
    - MLflow server running
    - RabbitMQ server running
    """
    print(f"Testing ExperimentHub API at {base_url}")
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test available experiments endpoint
    print("\nTesting available experiments endpoint...")
    try:
        response = requests.get(f"{base_url}/available_experiments")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test creating an experiment
    experiment_type = "time_series"
    experiment_name = "api_test_experiment"
    
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
        }
    }
    
    print(f"\nCreating experiment '{experiment_name}' of type '{experiment_type}'...")
    try:
        response = requests.post(
            f"{base_url}/create_experiment/{experiment_name}/{experiment_type}",
            json=config
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Wait for training to complete
    print("\nWaiting for training to complete...")
    time.sleep(10)
    
    # Test prediction
    print("\nTesting prediction...")
    try:
        response = requests.post(f"{base_url}/{experiment_name}/predict")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Prediction successful!")
            print(f"Response: {response.text}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nAPI test completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ExperimentHub API endpoints")
    parser.add_argument("--base-url", default="http://localhost:8010", help="Base URL of the ExperimentHub API")
    args = parser.parse_args()
    
    main(args.base_url)
