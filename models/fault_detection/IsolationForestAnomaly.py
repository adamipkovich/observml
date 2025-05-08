
import pickle
import mlflow

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots


class IsolationForestAnomaly(BaseEstimator):

    _model = None
    _data_sc = None
    _figs: dict[str, go.Figure] = {}

    def __init__(self, *, n_estimators : int  = 100, contamination = "auto", random_state = 0):
        """Isolation Forest model for fault detection
        """
        
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self._data_sc = StandardScaler()
        self._is_fitted = False
        
        
        
    def fit(self, X,y=None):
        self.scheme = X.columns
        n_data = self._data_sc.fit_transform(X)
        n_data = pd.DataFrame(n_data, columns=X.columns)
        ## TODO: gridsearch...
        # Detect outliers
        self._model = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, random_state=self.random_state).fit(n_data)
        self.anomalies = self._model.predict(n_data) == -1

        score = self._model.decision_function(n_data)

        self._figs['performance'] = make_subplots(rows=1, cols=1, subplot_titles=("Score"))
        counts, bins = np.histogram(score, bins=50)
        bins = 0.5 * (bins[:-1] + bins[1:])
        self._figs['performance'].add_trace(go.Bar(x = bins, y = counts, name = "Scores"), row=1, col=1)
        self._figs['performance'].update_layout(title_text='Isolation Forest Score Distribution')
        self._figs['performance'].update_xaxes(title_text='Score')
        self._figs['performance'].update_yaxes(title_text='Count')

        
        for i, col in enumerate(self.scheme):
            self._figs[f'{col}_outliers'] = make_subplots(rows=1, cols=1, subplot_titles=(f"Outliers for {col}",))
            self._figs[f'{col}_outliers'].add_trace(go.Scatter(x=X.index, y=X.loc[:, col], mode='lines', name=col, line=dict(color='blue')), row=1, col=1)
            self._figs[f'{col}_outliers'].add_trace(go.Scatter(x=X.index[self.anomalies], y=X.iloc[self.anomalies, i], mode='markers', name='Outliers', marker=dict(color='red')), row=1, col=1)
            self._figs[f'{col}_outliers'].update_xaxes(title_text='Index', row=1, col=1)
            self._figs[f'{col}_outliers'].update_yaxes(title_text=col, row=1, col=1)
            self._figs[f'{col}_outliers'].update_layout(showlegend=True)

        
        self.update_predict(X, reset_fig=True, update_fig=False)

        self._is_fitted_ = True
        return self



    def predict(self, X):

        n_data = self._data_sc.transform(X)
        n_data = pd.DataFrame(n_data, columns=X.columns)
        nX = X.copy()
        nX["outliers"] = self._model.predict(n_data)
        return nX
    
    def update_predict(self,data, reset_fig = False, update_fig = True):
        X = data
        if reset_fig:
            for i, col in enumerate(self.scheme):
                self._figs[f'{col}_predict'] = make_subplots(rows=1, cols=1, subplot_titles=(f"Outlier Prediction for {col}",))
                self._figs[f'{col}_predict'].update_xaxes(title_text='Index', row=1, col=1)
                self._figs[f'{col}_predict'].update_yaxes(title_text=col, row=1, col=1)
                self._figs[f'{col}_predict'].update_layout( showlegend=True)
        if update_fig:
            for i, col in enumerate(self.scheme):
                self._figs[f'{col}_predict'].add_trace(go.Scatter(x=X.index, y=X.loc[:, col], mode='lines', name=col, line=dict(color='blue')), row=1, col=1)
                self._figs[f'{col}_predict'].add_trace(go.Scatter(x=X.index[data["outliers"] == -1], y=X.loc[data["outliers"] == -1, col], mode='markers', name='Outliers', marker=dict(color='red')), row=1, col=1)
            


if __name__ == "__main__":

    data = pd.read_excel("./data/pump_train.xlsx")

    data["ds"] = pd.to_datetime(data["ds"])
    data.set_index("ds", inplace=True)
    model = IsolationForestAnomaly()
    model.fit(data)
    n_data = pd.read_excel("./data/pump_test_0.xlsx")
    n_data["ds"] = pd.to_datetime(n_data["ds"])
    n_data.set_index("ds", inplace=True)
    y = model.predict(n_data)
    model.update_predict(y)
    model._figs["performance"].show()
    model._figs["sensor_00_outliers"].show()
    model._figs["sensor_00_predict"].show()
    print("Done!")


