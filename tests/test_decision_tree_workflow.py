import api_commands
import yaml
import pandas as pd

# Load the DecisionTree configuration
with open('configs/detect/DecisionTree.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load the training data
data = pd.read_excel('./data/detect_train.xlsx')
data_json = data.to_json()

# Print some information
print("Configuration:")
print(config)
print("\nData shape:", data.shape)
print("Data columns:", data.columns.tolist())

# Train the model
print("\nTraining model...")
response = api_commands.train(
    cfg=config,
    url="http://localhost:8010",  # Use the port where API is running
    model_name="detect",       # Custom name for this model
    data=data_json,
    rabbit_host="localhost",
    rabbit_port="5672"            # Using default RabbitMQ port
)

# Check the response
print(f"\nResponse: {response.content}")
