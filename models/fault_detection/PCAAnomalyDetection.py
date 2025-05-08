from pca import pca
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import pickle
import mlflow
import logging

def plot_model(model, plot :str = "biplot"):
        return model.plot_model(plot)

def save_model(model, path : str = "model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path : str = "model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    

class PCAAnomalyDetection(BaseEstimator):
    _model : pca = None
    param : dict = None
    _std_scaler : StandardScaler = None
    _figs : dict[str, go.Figure] = {}
    outliers = None
    def __init__(self,  *, alpha = 0.05, detect_outliers = ['ht2', 'spe'], n_components = 0.95, normalize = True):

        self.alpha = alpha
        self.detect_outliers = detect_outliers
        self.n_components = n_components
        self.normalize = normalize
        self._model = pca(alpha=self.alpha,
                          detect_outliers=self.detect_outliers,
                          n_components=self.n_components,
                          normalize=self.normalize)
        #self._std_scaler = StandardScaler()
        
    def fit(self, X, y=None):

        self.scheme = X.columns
        #sc_X = self._std_scaler.fit_transform(X)
        self._model.fit_transform(X)
        #self.anomalies = self.results['outliers']['y_bool']
        self.ds = X.index
        self._figs["performance"] = self.plot_model("variance_ratio")
        self._figs["biplot"] = self.plot_model("biplot")
        self._figs["outliers"] = self.plot_model("outliers")

        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        #sc_x = self._std_scaler.transform(X)
        self._model.transform(X)
        self.ds = self.ds.union(X.index)
        self._model.results["PC"].set_index(self.ds, inplace=True)
        self._model.results['outliers']['y_bool'].index = self.ds
        self._model.results['outliers']['y_bool_spe'].index = self.ds
        self._figs["performance"] = self.plot_model("variance_ratio")   
        self._figs["biplot"] = self.plot_model("biplot")
        self._figs["outliers"] = self.plot_model("outliers")

        nX = X.copy()
        nX["outliers"] = self._model.results['outliers']['y_bool'].iloc[-X.shape[0]:]
        return nX
    
    def plot_model(self, plot :str = "outliers", fig = None, row = 1, col = 1):
        if plot == "biplot":
            hotelling_outliers = self._model.results['outliers']['y_bool']
            spe_outliers = self._model.results['outliers']['y_bool_spe']

            fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

            fig.add_trace(go.Scatter(x = self._model.results["PC"].loc[np.logical_and(hotelling_outliers == False, spe_outliers == False), "PC1"],
                                    y = self._model.results["PC"].loc[np.logical_and(hotelling_outliers == False, spe_outliers == False), "PC2"], 
                                    mode = "markers",
                                    marker_color="green",
                                    name = "Inliers", xaxis="x2", yaxis="y2"))

            fig.add_trace(go.Scatter(x = self._model.results["PC"].loc[np.logical_and(hotelling_outliers == True, spe_outliers == False), "PC1"],
                           y = self._model.results["PC"].loc[np.logical_and(hotelling_outliers == True, spe_outliers == False), "PC2"],
                             mode = "markers",
                             marker_symbol = "x", marker_color="blue", name = "Hotelling's T2 Outliers", xaxis="x2", yaxis="y2"))

            fig.add_trace(go.Scatter(x = self._model.results["PC"].loc[np.logical_and(hotelling_outliers == False, spe_outliers == True), "PC1"],
                           y = self._model.results["PC"].loc[np.logical_and(hotelling_outliers == False, spe_outliers == True), "PC2"],
                             mode = "markers",
                             marker_symbol = "x", marker_color="red", name = "SPE Outliers", xaxis="x2", yaxis="y2"))

            fig.add_trace(go.Scatter(x = self._model.results["PC"].loc[np.logical_and(hotelling_outliers == True, spe_outliers == True), "PC1"],
                           y = self._model.results["PC"].loc[np.logical_and(hotelling_outliers == True, spe_outliers == True), "PC2"],
                             mode = "markers",
                             marker_symbol = "x", marker_color="purple", name = "Hotelling and SPE Outliers", xaxis="x2", yaxis="y2"))

            fig.data[0].update(xaxis="x2", yaxis="y2")
            fig.data[1].update(xaxis="x2", yaxis="y2")
            fig.data[2].update(xaxis="x2", yaxis="y2")
            fig.data[3].update(xaxis="x2", yaxis="y2")
            
            fig.add_trace(go.Scatter(x =self._model.results["loadings"].loc["PC1", :], y = self._model.results["loadings"].loc["PC2", :], mode = "markers", name="Loadings", marker_symbol = "triangle-up", marker_color = "grey"))

            #fig.update_layout(xaxis = "x2", yaxis =  )
            labels = self._model.results["PC"].columns.tolist()
            loadings = self._model.results["loadings"] ## pcs are rows, features are columns


            for i, feature in enumerate(loadings.columns[0:len(labels)]):
                fig.add_annotation(
                    ax=0, ay=0,
                    axref="x", ayref="y",
                    x=loadings.loc["PC1", feature],
                    y=loadings.loc["PC2", feature],
                    showarrow=True,
                    arrowsize=2,
                    arrowhead=2,
                    xanchor="right",
                    yanchor="top"
                )
                fig.add_annotation(
                    x=loadings.loc["PC1", feature],
                    y=loadings.loc["PC2", feature],
                    axref="x", ayref="y",
                    ax=loadings.loc["PC1", feature], ay=loadings.loc["PC2", feature],
                    xanchor="center",
                    yanchor="bottom",
                    text=self.scheme[i],
                )
            fig.update_layout(title_text='Outliers',
                  yaxis2=dict(title='PC2 Scores', spikemode='toaxis', spikesnap='cursor'), 
                  yaxis=dict(title='PC2 Loadings', range=[min(1.1*loadings.iloc[1, 0:len(labels)].min(), 0),max(1.1*loadings.iloc[1, 0:len(labels)].max(), 0)]),
                  xaxis2 =dict(title='PC1 Scores', spikemode='across', spikesnap='cursor'),
                  xaxis=dict(title='PC1 Loadings', position = 1, anchor='free', overlaying='x', side='top',spikemode='across', spikesnap='cursor' , range=[min(1.1*loadings.iloc[0, 0:len(labels)].min(), 0), max(1.1*loadings.iloc[0, 0:len(labels)].max(), 0 )]) 
                 )
            
            return fig
    
        elif plot == "variance_ratio":
            if fig is None:
                fig = make_subplots(rows=row, cols=col)
            
            labels = self._model.results["PC"].columns.tolist()
            fig.add_trace(go.Bar(x = labels, y =  self._model.results["variance_ratio"][0:len(labels)], name = "Explained Variance Ratio"), row=row, col=col)
            fig.add_trace(go.Line(x =labels , y = self._model.results["explained_var"][0:len(labels)], name = "Cumulative Explained Variance"), row=row, col=col)        
            fig.update_yaxes(title_text="Explained Variance", row=row, col=col)
            fig.update_xaxes(title_text = "PCs", row=row, col=col)
            #fig.update_layout(title = "Explained Variance", xaxis_title = "PCs", yaxis_title = "Explained Variance")
            return fig

        elif plot =="outliers":

            fig = make_subplots(rows=2, cols=1)
            hotelling_outliers = self._model.results['outliers']['y_bool']
            spe_outliers = self._model.results['outliers']['y_bool_spe']
            PC = self._model.results["PC"]  
            #PC.reset_index(drop=True, inplace=True)

            ind = self._model.results["PC"].index #self._model.results["PC"].reset_index(drop=True).index ## pca model replaces index to "mapped"
            inliner_fig = go.Scatter(x = ind.to_series().loc[np.logical_and(hotelling_outliers == False, spe_outliers == False)],
                                    y = PC.loc[np.logical_and(hotelling_outliers == False, spe_outliers == False), "PC1"], 
                                    mode = "markers",
                                    marker_color="green",
                                    name = "Inliers")
            fig.add_trace(inliner_fig, col = 1, row = 1)

            inliner_fig = go.Scatter(x = ind.to_series().loc[np.logical_and(hotelling_outliers == True, spe_outliers == False)],
                                    y = PC.loc[np.logical_and(hotelling_outliers == True, spe_outliers == False), "PC1"], 
                                    mode = "markers",
                                    marker_color="blue",
                                    marker_symbol = "x",
                                    name =  "Hotelling's T2 Outliers")
            fig.add_trace(inliner_fig, col = 1, row = 1)

            inliner_fig = go.Scatter(x = ind.to_series().loc[np.logical_and(hotelling_outliers == False, spe_outliers == True)],
                                    y = PC.loc[np.logical_and(hotelling_outliers == False, spe_outliers == True), "PC1"], 
                                    mode = "markers",
                                    marker_color="red",
                                    marker_symbol = "x",
                                    name =  "SPE Outliers")

            fig.add_trace(inliner_fig, col = 1, row = 1)

            inliner_fig = go.Scatter(x = ind.to_series().loc[np.logical_and(hotelling_outliers == True, spe_outliers == True)],
                                    y = PC.loc[np.logical_and(hotelling_outliers == True, spe_outliers == True), "PC1"], 
                                    mode = "markers",
                                    marker_color="purple",
                                    marker_symbol = "x",
                                    name =  "Hotelling and SPE Outliers")

            fig.add_trace(inliner_fig, col = 1, row = 1)
            fig.update_yaxes(title_text="PC Score", row=1, col=1)
            fig.update_xaxes(title_text = "PC1", row = 1, col=1)

            inliner_fig = go.Scatter(x = ind.to_series().loc[np.logical_and(hotelling_outliers == False, spe_outliers == False)],
                                    y = PC.loc[np.logical_and(hotelling_outliers == False, spe_outliers == False), "PC2"], 
                                    mode = "markers",
                                    marker_color="green",
                                    name = "Inliers")
            fig.add_trace(inliner_fig, col = 1, row = 2)

            inliner_fig = go.Scatter(x = ind.to_series().loc[np.logical_and(hotelling_outliers == True, spe_outliers == False)],
                                    y = PC.loc[np.logical_and(hotelling_outliers == True, spe_outliers == False), "PC2"], 
                                    mode = "markers",
                                    marker_color="blue",
                                    marker_symbol = "x",
                                    name =  "Hotelling's T2 Outliers")
            fig.add_trace(inliner_fig, col = 1, row = 2)

            inliner_fig = go.Scatter(x = ind.to_series().loc[np.logical_and(hotelling_outliers == False, spe_outliers == True)],
                                    y = PC.loc[np.logical_and(hotelling_outliers == False, spe_outliers == True), "PC2"], 
                                    mode = "markers",
                                    marker_color="red",
                                    marker_symbol = "x",
                                    name =  "SPE Outliers")
            

            fig.add_trace(inliner_fig, col = 1, row = 2)

            inliner_fig = go.Scatter(x = ind.to_series().loc[np.logical_and(hotelling_outliers == True, spe_outliers == True)],
                                    y = PC.loc[np.logical_and(hotelling_outliers == True, spe_outliers == True), "PC2"], 
                                    mode = "markers",
                                    marker_color="purple",
                                    marker_symbol = "x",
                                    name =  "Hotelling and SPE Outliers")

            fig.add_trace(inliner_fig, col = 1, row = 2)

            fig.update_yaxes(title_text="PC Score", row=2, col=1)
            fig.update_xaxes(title_text = "PC2", row = 2, col=1)

            return fig
    
    def update_predict(self, X, reset_fig = False, update_fig = True):
        logging.info("PCA does not require figure update.")
        
    
if __name__ == "__main__":


    data = pd.read_excel("./data/pump_train.xlsx")
    data.set_index("ds", inplace=True)
    pca_model = PCAAnomalyDetection() 
    pca_model.fit(data)
    n_data = pd.read_excel("./data/pump_test.xlsx")
    
    n_data.set_index("ds", inplace=True)
    y = pca_model.predict(n_data)

    print("Done.")
