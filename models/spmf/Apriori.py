import os
from models.spmf.SPMFCore import SPMFCore
from sklearn.base import BaseEstimator
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

#FPGrowth_association_rules ++
class Apriori(BaseEstimator):
    _model = None
    _figs : dict[str, go.Figure] = {}

    def __init__(self, *, minsup ="10%", min_conf = "10%"):
        ## check if spmf.jar exists
        if not os.path.exists("spmf.jar"):
            raise FileNotFoundError("spmf.jar not found")
        
        self.minsup = minsup
        self.min_conf = min_conf
        self.model_name = "Apriori_association_rules"

    def fit(self, X: pd.DataFrame , y = None):
        self._model = SPMFCore(mode=self.model_name, minsup = self.minsup, minconf=self.min_conf)
        self.predict(X)
        return self

    def predict(self, X :pd.DataFrame):
        nX = self.reformat_data(X)
        self.result = self._model.run(nX)
        self._figs["Support"] = px.bar(self.result, x='Rule', y='Support',  title=f'Support with minimum support: {self.minsup}')
        self._figs["Confidence"] = px.bar(self.result, x='Rule', y='Confidence', title=f'Support with minimum support: {self.minsup}')

        self._figs["association_rules"] = go.Figure()

        edges = []
        for i in range(0, len(self.result)):
            edge =self.result["Rule"][i].split(" ==> ")
            edge[1] = edge[1][:-1]
            edges.append(edge)
        # get all edges
        G = nx.DiGraph()
        G.add_edges_from(edges)
        layout = nx.planar_layout(G) 
        layout = nx.rescale_layout_dict(layout, scale=0.3) 
        for node in G.nodes():
            self._figs["association_rules"].add_trace(go.Scatter(x=[layout[node][0]], y=[layout[node][1]], mode='markers+text', text=node, textposition="top center", marker = dict(size=40, color="blue"), showlegend=False, hoverinfo='skip'))
        for edge in G.edges():
            x0 = layout[edge[0]][0]
            y0 = layout[edge[0]][1]
            x1 = layout[edge[1]][0]
            y1 = layout[edge[1]][1] 
            self._figs["association_rules"].add_annotation(x=x0, y=y0, ax=x1, ay = y1, axref="x", ayref="y", text='', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#636363')

        self._figs["association_rules"].update_layout(showlegend=False)
        self._figs["association_rules"].update_xaxes(showticklabels=False)
        self._figs["association_rules"].update_yaxes(showticklabels=False)
        self._figs["association_rules"].update_layout(title='Association Rules Connections')
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
    tkr = Apriori()
    tkr.fit(X)
    tkr._figs["Support"].show()
    tkr._figs["Confidence"].show()
    tkr._figs["association_rules"].show()
    print("Done!")