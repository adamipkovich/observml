import pandas as pd
from api_commands import train_models, retrain, predict


train_models(conf_path="./configs",
                    conf_name="config",
                    data_path="data.yaml", 
                    url="http://localhost:8010",
                    rabbit_host="localhost",
                    rabbit_port="5100", 
                    user="guest",
                    password="guest"
                    )

for i in range(8):
    data = pd.read_excel(f"./data/pump_test_{i}.xlsx")
    data.reset_index(inplace=True, drop=True)
    data = data.to_json()
    predict(url="http://localhost:8010", model_name = "pump", data = data, rhost = "localhost", rport = "5100")

#retrain("detect")