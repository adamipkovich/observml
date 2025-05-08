import os
from models.spmf.SPMFCore import SPMFCore
from sklearn.base import BaseEstimator
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

class TopKRules(BaseEstimator):
    _model = None
    _figs : dict[str, go.Figure] = {}

    def __init__(self, *, k = 10, minconf_ar = 0.8):
        ## check if spmf.jar exists
        if not os.path.exists("spmf.jar"):
            raise FileNotFoundError("spmf.jar not found")
        
        self.k = k
        self.minconf_ar = minconf_ar
        self.model_name = "TopKRules"

    def fit(self, X: pd.DataFrame , y = None):
        
        self._model = SPMFCore(mode=self.model_name, k=self.k, minconf_ar=self.minconf_ar)
        self.predict(X)    

        return self

    def predict(self, X :pd.DataFrame):
        self.result = self._model.run(X)
        
        self._figs["Support"] = px.bar(self.result, x='Rule', y='Support',  title=f'Support with minimum support: {self.minconf_ar*100} %')
        self._figs["Confidence"] = px.bar(self.result, x='Rule', y='Confidence', title=f'Support with minimum support: {self.minconf_ar*100} %')

        self._figs["association_rules"] = go.Figure()

        edges = []
        for i in range(0, len(self.result)):
            edge =self.result["Rule"][i].split("  ==> ")
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

        ## TODO: add weight to graphs (support!)

        return self.result

if __name__ == "__main__":
    X = pd.read_csv("./data/traces_csoft_oper.csv")
    tkr = TopKRules()
    tkr.fit(X)
    tkr._figs["Support"].show()
    tkr._figs["Confidence"].show()
    tkr._figs["association_rules"].show()
    print("Done!")