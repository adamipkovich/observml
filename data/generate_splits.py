from tools import read_data
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

data = read_data("./data/detect_dataset.xlsx", engine="pandas")
data["ds"] = pd.date_range(start="1/1/2018", periods=data.shape[0], freq="H")
train, test= train_test_split(data, test_size = 0.3, shuffle=False)
train.to_excel("./data/detect_train.xlsx", index = False)


test_sets = [test]
subdiv = 3

for i in range(subdiv):
    n_test = list()
    for each in test_sets:
        t1, t2 = train_test_split(each, test_size = 0.5, shuffle=False)
        n_test.append(t1)
        n_test.append(t2)
    test_sets = n_test      

for i, each in enumerate(test_sets):
    each.to_excel( f"./data/detect_test_{i}.xlsx", index = False)

data = read_data("./data/time_series_data.xlsx", engine="pandas")
train, test= train_test_split(data, test_size = 0.4, shuffle=False)
train.to_excel("./data/time_series_train.xlsx", index = False)
test_sets = [test]
subdiv = 3
for i in range(subdiv):
    n_test = list()
    for each in test_sets:
        t1, t2 = train_test_split(each, test_size = 0.5, shuffle=False)
        n_test.append(t1)
        n_test.append(t2)
    test_sets = n_test        

for i, each in enumerate(test_sets):
    each.to_excel( f"./data/time_series_test_{i}.xlsx", index = False)


data = read_data("./data/pump_sensor_data.xlsx", engine="pandas")
data["ds"] = pd.date_range(start="1/1/2018", periods=data.shape[0], freq="H")
train, test= train_test_split(data, test_size = 0.4, shuffle=False)
train.to_excel("./data/pump_train.xlsx", index = False)

test_sets = [test]
subdiv = 3

for i in range(subdiv):
    n_test = list()
    for each in test_sets:
        t1, t2 = train_test_split(each, test_size = 0.5, shuffle=False)
        n_test.append(t1)
        n_test.append(t2)
    test_sets = n_test        


for i, each in enumerate(test_sets):
    each.to_excel( f"./data/pump_test_{i}.xlsx", index = False)


data = read_data("./data/hmm_fault_diagnosis_dataset.csv", engine="pandas")
train, test= train_test_split(data, test_size = 0.1, shuffle=False)
train.to_excel("./data/hmm_train.xlsx", index = False)
test.to_excel( "./data/hmm_test.xlsx", index = False)