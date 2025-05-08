"""Example for training a model using the API, with data injection."""

if __name__ == "__main__":
    from api_commands import compile_config, train
    import pandas as pd
    conf = compile_config(conf_path = "./configs", conf_name = "config") 
    for k in conf.keys():
        
        if conf[k]["create_model"]["model"] == "apriori" or conf[k]["create_model"]["model"] == "topk" or conf[k]["create_model"]["model"] == "cmspam":
                data = pd.read_csv("./data/traces_csoft_oper.csv")
                data = data.to_json()
        elif conf[k]["create_model"]["model"] == "heuristics": 
            data = pd.read_csv("./data/cable_head_mach_27.csv")
            data = data.to_json()
        else:
            data = pd.read_excel(f"./data/{k}_train.xlsx")
            data.reset_index(inplace=True, drop=True)
            data = data.to_json()

        rep = train(conf[k], url="http://localhost:8010", model_name = k, data = data, rabbit_host = "localhost", rabbit_port = "5100")