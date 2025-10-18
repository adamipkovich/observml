"""Example for testing a model using the API, with data injection."""

if __name__ == "__main__":
    from api_commands import predict
    import pandas as pd

    for i in range(8):
         data = pd.read_excel(f"./data/detect_test_{i}.xlsx")
         data.reset_index(inplace=True, drop=True)
         data = data.to_json()
         predict(url="http://localhost:8010", model_name = "detect", data = data, rhost = "localhost", rport = "5100")

    #data = pd.read_csv(f"./data/torony_test.csv")
    #data.reset_index(inplace=True, drop=True)
    #data = data.to_json()
    # predict(url="http://localhost:8010", model_name = "iszap_isofor", data = data, rhost = "localhost", rport = "5100")
    # predict(url="http://localhost:8010", model_name = "iszap_ssa", data = data, rhost = "localhost", rport = "5100")
    # predict(url="http://localhost:8010", model_name = "iszap_pca", data = data, rhost = "localhost", rport = "5100")
    # predict(url="http://localhost:8010", model_name = "iszap_nb", data = data, rhost = "localhost", rport = "5100")
    # predict(url="http://localhost:8010", model_name = "iszap_autoenc", data = data, rhost = "localhost", rport = "5100")
    # predict(url="http://localhost:8010", model_name = "iszap_mc", data = data, rhost = "localhost", rport = "5100")
    # predict(url="http://localhost:8010", model_name = "iszap_elliptic", data = data, rhost = "localhost", rport = "5100")
    # predict(url="http://localhost:8010", model_name = "iszap_dt", data = data, rhost = "localhost", rport = "5100")
    #predict(url="http://localhost:8010", model_name = "iszap_hmm", data = data, rhost = "localhost", rport = "5100")
    # predict(url="http://localhost:8010", model_name = "iszap_pr", data = data, rhost = "localhost", rport = "5100")