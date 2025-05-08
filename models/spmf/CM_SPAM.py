import os
from models.spmf.SPMFCore import SPMFCore
from sklearn.base import BaseEstimator
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

#FPGrowth_association_rules ++
class CMSPAM(BaseEstimator):
    _model = None
    _figs : dict[str, go.Figure] = {}

    def __init__(self, *, minsup ="50%", min_len = 1, max_len = 10, req_items = list(), max_gap = 3):
        ## check if spmf.jar exists
        if not os.path.exists("spmf.jar"):
            raise FileNotFoundError("spmf.jar not found")
        
        self.minsup = minsup
        self.min_len = min_len
        self.max_len = max_len

        if len(req_items) == 0:
            req_items = '""'
        else:
            req_items = "".join([str(x) + "," for x in req_items][0:-1])

        
        self.req_items = req_items
        self.max_gap = max_gap  
        
        self.model_name = "CM-SPAM"

    def fit(self, X: pd.DataFrame , y = None):
        
        self._model = SPMFCore(mode=self.model_name, minsup = self.minsup, minlen=self.min_len, maxlen=self.max_len, req_items=self.req_items, max_gap=self.max_gap)
        self.predict(X)
        return self

    def predict(self, X :pd.DataFrame):
        nX = self.reformat_data(X)
        self.result = self._model.run(nX)
        self._figs["Support"] = px.bar(self.result, x='Rule', y='Support',  title=f'Support with minimum support: {self.minsup} %')
        return self.result
    
    def reformat_data(self, X:pd.DataFrame):
        for i in range(0, X.shape[0]):
            if X.loc[i][0][0] != "@":
                st = X.loc[i][0].split(" ")
                n_st = []
                for s in st:
                    if s != "":
                        n_st.append(s + " -1 ")
                X.loc[i][0] = "".join(n_st)[0:-2] + "2" 
                
        return X

if __name__ == "__main__":
    ## requires different format to topkrules ->> may need to reconfigure data
    X = pd.read_csv("./data/traces_csoft_oper.csv")
    tkr = CMSPAM()
    tkr.fit(X)
    tkr._figs["Support"].show()

    print("Done!")