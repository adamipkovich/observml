"""Example of a training script that uses the API to train a model, instead of CLI commands."""

if __name__ == "__main__":
    
    import pandas as pd
    import api_commands as api
    import pickle
    import yaml
    from yaml import SafeLoader

    api.train_models(conf_path="./configs",
                    conf_name="config",
                    data_path="data.yaml", 
                    url="http://localhost:8010",
                    rabbit_host="localhost",
                    rabbit_port="5100", 
                    user="guest",
                    password="guest")

