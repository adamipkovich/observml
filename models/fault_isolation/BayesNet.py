import pandas as pd
import numpy as np
#import bnlearn as bn
import mlflow
from mlflow.client import MlflowClient

import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

import distinctipy

class BayesNet(BaseEstimator):

    _model : BayesianNetwork = None
    _figs : dict[str, go.Figure] = {}

    def __init__(self, *, placeholder = None) -> None:
        pass

    def fit(self, X, y=None): 
        if "target" not in X.columns:
            raise Exception("target column not found in the data.")


        ds = X.index
        #
        X.reset_index(drop=True, inplace=True)
        nX = X.copy()

        #y = nX["target"]
        #nX = nX.drop(columns=["target"])

        self.scheme = nX.columns.to_list()
        self.scheme.remove("target")

        ## Binning continuous values
        for cols in self.scheme:
            nX[cols], _bins_edges = pd.qcut(nX[cols], q=100, duplicates='drop', retbins=True)

        ##
        train, test = train_test_split(nX, test_size=0.33, random_state=42)

        # Learn the structure of the Bayesian Network
        hc = HillClimbSearch(train)
        best_model = hc.estimate(scoring_method=BicScore(train))

        # Define the model with the learned structure
        self._model = BayesianNetwork(best_model.edges())

        # Train the model using Maximum Likelihood Estimation
        self._model.fit(train, estimator=MaximumLikelihoodEstimator)

        ## Prediction (for missing column, so for the target variable)
        nX.drop(columns=["target"], inplace=True)   
        y_hat = self._model.predict(nX)

        cm = confusion_matrix(X["target"], y_hat, normalize='true')
        self.classes_ = X["target"].unique()
        # Confusion Matrix
        self._figs['performance'] = make_subplots(rows=1, cols=1, subplot_titles=("Confusion Matrix",))
        self._figs['performance'].add_trace(go.Heatmap(z=cm, x=self.classes_, y=self.classes_), row=1, col=1)
        self._figs['performance'].update_layout(title_text='Confusion Matrix', width=500, height=500, showlegend=False)
        self._figs['performance'].update_xaxes(title_text='Predicted', row=1, col=1)
        self._figs['performance'].update_yaxes(title_text='Actual', row=1, col=1)

        colors = ["rgb" + str(c) for c in distinctipy.get_colors(X["target"].nunique())]


        self.colors = {}
        for i, cl in enumerate(X["target"].unique()):
            self.colors[cl] = colors[i]

        for j, col in enumerate(self.scheme):
                rX = X.loc[:, col]

                self._figs[f'{col}_classes'] = make_subplots(rows=1, cols=1, subplot_titles=("Classes",))
                self._figs[f'{col}_classes'].update_xaxes(title_text='Time', row=1, col=1)
                self._figs[f'{col}_classes'].update_yaxes(title_text= col, row=1, col=1)
                self._figs[f'{col}_classes'].update_layout(showlegend=False)

                for i, cl in enumerate(self.colors.keys()):
                    fig_d =  rX.loc[(y_hat == cl)["target"]]
                    ds = fig_d.index.to_list()
                    sub_col = self.colors[cl]
                    self._figs[f'{col}_classes'].add_trace(go.Scatter(x=ds, y=fig_d, mode='markers', name=str(cl), marker=dict(color=sub_col)), row=1, col=1)

        
        self.update_predict(X, reset_fig=True, update_fig=False)

        self._is_fitted = True
        return self

    
    def _score(self, y, y_hat):
        return confusion_matrix(y, y_hat)
        

    def predict(self, X):
        """Cannot be used without a target variable!!!"""
        if not  self._is_fitted:
            raise Exception("Model not fitted. Please fit the model first.")
        if "target" in X.columns:
            nX = X.drop(columns=["target"]).copy()
        else:
            nX = X.copy()
        # if "target" in X.columns:
        #     nX = X.drop(columns=["target"])
        # else:
        #     nX = X.copy()
        #cols = nX.columns
        #nX = self._std_scaler.transform(nX)
        #nX = pd.DataFrame(nX, columns=self.scheme)
        nX.reset_index(drop=True, inplace=True)
        for cols in self.scheme:
            nX[cols], _bins_edges = pd.qcut(nX[cols], q=100, duplicates='drop', retbins=True)
        y_hat = self._model.predict(nX)
        y_hat = y_hat.set_index(X.index, drop=True)
        X["y_pred"] = y_hat
        return X #, anomaly_ratio, MAE, MSE, RMSE, MAPE, mean_uncertainty

    def update_predict(self, X, reset_fig = False, update_fig = True):
        if reset_fig:
            for i, col in enumerate(self.scheme):
                self._figs[f'{col}_prediction'] = make_subplots(rows=1, cols=1, subplot_titles=("Prediction",))
                self._figs[f'{col}_prediction'].update_xaxes(title_text='Time', row=1, col=1)
                self._figs[f'{col}_prediction'].update_yaxes(title_text=col, row=1, col=1)
            
        if update_fig:
            for j, col in enumerate(self.scheme):
                rX = X.loc[:, col]
                for i, cl in enumerate(self.colors.keys()):
                    fig_d =  rX.loc[X["y_pred"] == cl]
                    ds = fig_d.index.to_list()
                    sub_col = self.colors[cl]
                    self._figs[f'{col}_prediction'].add_trace(go.Scatter(x=ds, y=fig_d, mode='markers', name=str(cl), marker=dict(color=sub_col)), row=1, col=1)

if __name__ == "__main__":
    
    data = pd.read_excel("./data/detect_train.xlsx")
    model = BayesNet()

    X = data.rename(columns={"Output (S)": "target", "ds": "ds"})
    X["ds"] = pd.to_datetime(X["ds"])
    X.set_index("ds", inplace=True)
    model.fit(X)
    model._figs["Ia_classes"].show()
    

    nX = pd.read_excel("./data/detect_test_0.xlsx")
    nX = data.rename(columns={"Output (S)": "target", "ds": "ds"})
    nX["ds"] = pd.to_datetime(nX["ds"])
    nX.set_index("ds", inplace=True)
    y = model.predict(nX)
    model.update_predict(y)

    model._figs["Ia_prediction"].show()
    

    #n_data = pd.read_excel("./data/detect_test.xlsx")
    print("Done!")