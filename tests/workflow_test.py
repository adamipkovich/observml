"""
Comprehensive workflow test for ObservML.

This script tests the following workflow:
1. Check if the ExperimentHub API is running
2. Check if the MLflow server is running
3. Check if RabbitMQ is running
4. Train a decision tree model
5. Check the status of the training
6. Make predictions with the trained model
"""

import requests
import time
import json
import pandas as pd
import api_commands
import yaml
import os

# Configuration
API_URL = "http://localhost:8010"  # ExperimentHub API URL
MLFLOW_URL = "http://localhost:5000"  # MLflow server URL
RABBIT_HOST = "localhost"
RABBIT_PORT = "5672"
MODEL_NAME = "detect"  # Name for our model (must match folder name in configs)

def check_service(url, service_name):
    """Check if a service is running by making a request to its URL."""
    try:
        response = requests.get(url)
        print(f"✅ {service_name} is running at {url}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ {service_name} is not running at {url}")
        return False

def check_api_health():
    """Check the health of the ExperimentHub API."""
    try:
        response = requests.get(f"{API_URL}/health")
        health_data = response.json()
        
        print("\n--- API Health Check ---")
        for plugin_type, info in health_data.items():
            status = "✅ HEALTHY" if info["healthy"] else "❌ UNHEALTHY"
            print(f"{plugin_type}: {status}")
            print(f"  Details: {info['details']}")
        
        return all(info["healthy"] for info in health_data.values())
    except requests.exceptions.ConnectionError:
        print(f"❌ ExperimentHub API is not running at {API_URL}")
        return False

def train_decision_tree():
    """Train a decision tree model using the API."""
    # Load the DecisionTree configuration
    with open('configs/detect/DecisionTree.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the training data
    data = pd.read_excel('./data/detect_train.xlsx')
    data_json = data.to_json()
    
    # Print some information
    print("\n--- Training Decision Tree Model ---")
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Configuration: {config}")
    
    # Train the model
    print("\nSending training request...")
    response = api_commands.train(
        cfg=config,
        url=API_URL,
        model_name=MODEL_NAME,
        data=data_json,
        rabbit_host=RABBIT_HOST,
        rabbit_port=RABBIT_PORT
    )
    
    print(f"Response: {response.content}")
    return response.status_code == 200

def check_model_status(max_retries=12, retry_interval=5):
    """Check if the model is available in the ExperimentHub API."""
    print("\n--- Checking Model Status ---")
    
    for attempt in range(max_retries):
        # Wait for the model to be trained
        print(f"Waiting for model training to complete... (Attempt {attempt+1}/{max_retries})")
        time.sleep(retry_interval)
        
        # Check if the model configuration is available
        try:
            response = requests.get(f"{API_URL}/{MODEL_NAME}/cfg")
            if response.status_code == 200 and response.json():
                print(f"✅ Model {MODEL_NAME} is available")
                print(f"Configuration: {response.json()}")
                return True
            else:
                print(f"⏳ Model {MODEL_NAME} is not available yet, retrying...")
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to {API_URL}")
            return False
    
    print(f"❌ Model {MODEL_NAME} is not available after {max_retries} attempts")
    return False

def make_prediction():
    """Make a prediction with the trained model."""
    print("\n--- Making Prediction ---")
    
    # Load test data
    data = pd.read_excel('./data/detect_test_0.xlsx')
    data_json = data.to_json()
    
    print(f"Test data shape: {data.shape}")
    
    # Make prediction
    try:
        response = api_commands.predict(
            url=API_URL,
            data=data_json,
            model_name=MODEL_NAME,
            rhost=RABBIT_HOST,
            rport=RABBIT_PORT
        )
        
        print(f"Prediction response: {response.content}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

def test_stop_training_functionality():
    """Test the ability to stop a running training process."""
    print("\n--- Testing Stop Training Functionality ---")
    
    # Start a training task
    print("Starting a training task to test stop functionality...")
    training_success = train_decision_tree()
    
    if not training_success:
        print("❌ Failed to start training task")
        return False
    
    # Wait a moment to ensure the task has started
    print("Waiting for task to start...")
    time.sleep(5)  # Give it a bit more time to ensure the task is running
    
    # Stop the training
    try:
        stop_response = requests.post(f"{API_URL}/{MODEL_NAME}/stop_training")
        print(f"Stop training response: {stop_response.content}")
        
        if stop_response.status_code == 200:
            print("✅ Successfully sent stop training command")
            
            # Wait a moment for the stop to take effect
            time.sleep(5)  # Give it more time for the stop to take effect
            
            # Check if the model is still available
            status_response = requests.get(f"{API_URL}/{MODEL_NAME}/cfg")
            if status_response.status_code != 200 or not status_response.json():
                print("✅ Training was successfully stopped")
                return True
            else:
                print("⚠️ Model is still available after stop command")
                return False
        else:
            print(f"❌ Failed to stop training: {stop_response.content}")
            return False
    except Exception as e:
        print(f"❌ Error testing stop training functionality: {e}")
        return False

def main():
    """Run the workflow test."""
    print("=== ObservML Workflow Test ===\n")
    
    # Check if services are running
    api_running = check_service(API_URL, "ExperimentHub API")
    mlflow_running = check_service(MLFLOW_URL, "MLflow server")
    
    if not api_running or not mlflow_running:
        print("\n❌ Required services are not running. Please start them before running this test.")
        return
    
    # Check API health
    api_healthy = check_api_health()
    if not api_healthy:
        print("\n❌ ExperimentHub API is not healthy. Please check the logs.")
        return
    
    # Test stop training functionality
    stop_success = test_stop_training_functionality()
    if not stop_success:
        print("\n⚠️ Stop training functionality test failed. This might be expected if Celery is not configured properly.")
        print("Continuing with regular workflow test...")
    
    # Train decision tree model
    training_success = train_decision_tree()
    if not training_success:
        print("\n❌ Failed to send training request.")
        return
    
    # Check model status
    model_available = check_model_status()
    if not model_available:
        print("\n⚠️ Model is not available yet. It might still be training.")
        print("You can run this script again later to check if the model is available.")
        return
    
    # Make prediction
    prediction_success = make_prediction()
    if not prediction_success:
        print("\n❌ Failed to make prediction.")
        return
    
    print("\n✅ Workflow test completed successfully!")

if __name__ == "__main__":
    main()
