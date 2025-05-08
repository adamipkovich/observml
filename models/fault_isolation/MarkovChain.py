import numpy as np
import pandas as pd
from mchmm import MarkovChain
import mlflow

from mlflow.client import MlflowClient
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
import distinctipy


class MarkovChainModel(BaseEstimator):
    _model : MarkovChain = None
    _figs : dict[str, go.Figure] = {}

    def __init__(self, *, placeholder = None):
            """Hidden Markov Model for hidden states detection."""
            pass
    

    def fit(self, X, y=None):
        if "target" not in X.columns:
            raise Exception("target column not found in the data.")
        y = X["target"]
        X.drop(columns=["target"], inplace=True)
        self.no_states = y.unique().shape[0]
        #y = y.astype(str)
        self._model = MarkovChain().from_data(y)
        #     self._figs['state'] = self._model.graph_make(
        #           format="png",
        #           graph_attr=[("rankdir", "LR")],
        #           node_attr=[("fontname", "Roboto bold"), ("fontsize", "20")],
        #           edge_attr=[("fontname", "Iosevka"), ("fontsize", "12")]
        #     )        
        self.state = self._model.seq    
        transition_matrix = self._model.observed_p_matrix
        self._figs['performance'] = make_subplots(rows=1, cols=1, subplot_titles=("Transition Matrix",))
        self._figs['performance'].add_trace(go.Heatmap(z=transition_matrix, x=self._model.states, y=self._model.states))

        #add colorbar   
        self._figs['performance'].update_xaxes(title_text="p value-based transition")
        self._figs['performance'].update_yaxes(title_text="p value-based transition")
        self._figs['performance'].update_layout(title_text="Transition Matrix", showlegend=False)
        
        self.scheme = X.columns.tolist()

        colors = ["rgb" + str(c) for c in distinctipy.get_colors(y.nunique())]
        self.colors = {}
        for i, cl in enumerate(np.unique(self._model.states)):
            self.colors[cl] = colors[i]

        for j, col in enumerate(self.scheme):
                rX = X.loc[:, col]

                self._figs[f'{col}_classes'] = make_subplots(rows=1, cols=1, subplot_titles=("Classes",))
                self._figs[f'{col}_classes'].update_xaxes(title_text='Time', row=1, col=1)
                self._figs[f'{col}_classes'].update_yaxes(title_text= col, row=1, col=1)
                self._figs[f'{col}_classes'].update_layout(showlegend=True)

                for i, cl in enumerate(self.colors.keys()):
                    fig_d =  rX.loc[self.state == cl]
                    ds = fig_d.index.to_list()
                    sub_col = self.colors[cl]
                    self._figs[f'{col}_classes'].add_trace(go.Scatter(x=ds, y=fig_d, mode='markers', name=str(cl), marker=dict(color=sub_col)), row=1, col=1)

        
        self.update_predict(X, reset_fig=True, update_fig=False)

        self._is_fitted = True
        return self

    
    def _score(self, y, y_hat):
            logging.log(logging.INFO, "Does not have score feature.")
            return None
        

    def predict(self, X):
        if "target" in X.columns:
            nX = X.drop(columns=["target"])
        else:
            nX = X.copy()
        #y = self._model.simulate(self.state.shape[0] + X.shape[0])[0]
        #nX = X.copy()
        #nX["y_pred"] = y[-X.shape[0]:]
        y = self._model.simulate(X.shape[0])[0]
        nX = X.copy()
        nX["y_pred"] = y
        return nX
    

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
    model = MarkovChainModel()

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
    model.update_predict(y, reset_fig = False, update_fig = True)

    model._figs["Ia_prediction"].show()
    

    #n_data = pd.read_excel("./data/detect_test.xlsx")
    print("Done!")