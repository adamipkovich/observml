import numpy as np
import pandas as pd
from hmmlearn import hmm
import mlflow

from mlflow.client import MlflowClient

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

import networkx as nx
import distinctipy

class HMM(BaseEstimator):
    _model : hmm = None
    _figs : dict[str, go.Figure] = {}

    def __init__(self, *, n_iter = 1000, covariance_type="diag", n_mix=10):
            """Hidden Markov Model for hidden states detection.
            :param data: input data
            :param params: parameters
            - n_iter: number of iterations

            :return: None"""
            self.n_iter = n_iter
            self.covariance_type = covariance_type
            self.n_mix = n_mix
            self._std_scaler = StandardScaler()
  

    def fit(self, X, y=None):
            if "target" not in X.columns:
                raise Exception("state column not found in the data.")
            no_states = X['target'].unique().shape[0]
            self.no_states = no_states

            nX = X.drop(columns=["target"])
            self.scheme = nX.columns

            nX = self._std_scaler.fit_transform(nX)
            nX = pd.DataFrame(nX, columns=self.scheme, index = X.index)

            y = X['target']

            self._model = hmm.GMMHMM(n_components=no_states, covariance_type=self.covariance_type, n_iter=self.n_iter, n_mix=self.n_mix)

            self._model.fit(nX)
            self.state = self._model.predict(nX)
            
            self.cf_matrix = confusion_matrix(y, self.state, normalize='true')
             # Confusion Matrix
            self._figs['performance'] = make_subplots(rows=1, cols=1, subplot_titles=("Confusion Matrix",))
            self._figs['performance'].add_trace(go.Heatmap(z=self.cf_matrix, x=np.unique(self.state), y=np.unique(self.state)), row=1, col=1)
            self._figs['performance'].update_layout(title_text='Confusion Matrix', width=500, height=500, showlegend=True)
            self._figs['performance'].update_xaxes(title_text='Predicted', row=1, col=1)
            self._figs['performance'].update_yaxes(title_text='Actual', row=1, col=1)

            colors = ["rgb" + str(c) for c in distinctipy.get_colors(X["target"].nunique())]
            self.colors = {}
            for i, cl in enumerate(y.unique()):
                self.colors[cl] = colors[i]

            for j, col in enumerate(self.scheme):
                    rX = nX.loc[:, col]

                    self._figs[f'{col}_classes'] = make_subplots(rows=1, cols=1, subplot_titles=("Classes",))
                    self._figs[f'{col}_classes'].update_xaxes(title_text='Time', row=1, col=1)
                    self._figs[f'{col}_classes'].update_yaxes(title_text= col, row=1, col=1)
                    self._figs[f'{col}_classes'].update_layout(showlegend=False)

                    for i, cl in enumerate(self.colors.keys()):
                        fig_d =  rX.loc[self.state == cl]
                        ds = fig_d.index.to_list()
                        sub_col = self.colors[cl]
                        self._figs[f'{col}_classes'].add_trace(go.Scatter(x=ds, y=fig_d, mode='markers', name=str(cl), marker=dict(color=sub_col)), row=1, col=1)
                        self._figs[f'{col}_classes'].update_layout(showlegend=True)
            
            self.update_predict(X, reset_fig=True, update_fig=False)
            self._is_fitted = True
            return self

    
    def _score(self, y, y_hat):
            return confusion_matrix(y, y_hat)
        

    def predict(self, X):
        if not self._is_fitted:
            raise Exception("Model not fitted. Please fit the model first.")
        
        if "target" in X.columns:
            nX = X.drop(columns=["target"])
        else:
            nX = X.copy()

        cols = nX.columns
        nX = self._std_scaler.transform(nX)
        nX = pd.DataFrame(nX, columns=cols, index = X.index)
        y_hat = self._model.predict(nX)
        X["y_pred"] = y_hat
        return X
    

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


# if __name__ == "__main__":

#     data = pd.read_excel("./data/detect_train.xlsx")
#     X = data.rename(columns={"Output (S)": "target", "ds": "ds"})
#     X["ds"] = pd.to_datetime(X["ds"])
#     X.set_index("ds", inplace=True)
#     model = HMM()
#     model.fit(X)

#     nX = pd.read_excel("./data/detect_test_0.xlsx")
#     nX = data.rename(columns={"Output (S)": "target", "ds": "ds"})
#     nX["ds"] = pd.to_datetime(nX["ds"])
#     nX.set_index("ds", inplace=True)
#     y = model.predict(nX)

#     model.update_predict(y, reset_fig = False, update_fig = True)

#     model._figs["performance"].show()
#     model._figs["Ia_classes"].show()
#     model._figs["Ia_prediction"].show()
#     print("Done!")