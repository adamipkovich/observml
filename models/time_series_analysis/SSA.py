import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from warnings import warn
import mlflow

import plotly
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

from pyts.decomposition import SingularSpectrumAnalysis


class SSAAnomalyDetection(BaseEstimator):
    _model : SingularSpectrumAnalysis = None
    _figs : dict[str, go.Figure] = dict().copy() 

    def __init__(self, * , window_size = 10, lower_frequency_bound = 0.05, lower_frequency_contribution = 0.975, threshold = 3):
        self.window_size = window_size
        self.lower_frequency_bound = lower_frequency_bound
        self.lower_frequency_contribution = lower_frequency_contribution
        self.threshold = threshold
        self._model = SingularSpectrumAnalysis(window_size=self.window_size,
                                               groups='auto',
                                               lower_frequency_bound=0.1,
                                               lower_frequency_contribution=0.95,)
        self.data_sc = StandardScaler()
        self.error_sc = StandardScaler()

    def fit(self, X, y=None):
        assert 'ds' in X.columns, "ds column must be present in the data"
        assert 'y' in X.columns, "y column must be present in the data"

        n_data = self.data_sc.fit_transform(X.loc[:, ['y']])
        X_ssa = self._model.fit_transform(n_data.transpose())
        X_ssa = np.reshape(X_ssa, (X_ssa.shape[1], X_ssa.shape[2]))

        self._std_model = StandardScaler()
        res_std = self._std_model.fit_transform(np.reshape(X_ssa[2, :], (-1, 1)))
        self.anomalies = np.abs(res_std) > self.threshold 

        self._figs["performance"] = make_subplots(rows=3, cols=1)
        #self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = n_data, mode = "lines", marker_color = "black", name = "Data"), row=1, col=1)
        self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X_ssa[0, :], mode = "lines", marker_color = "blue"), row=1, col=1)
        self._figs["performance"].update_xaxes(title_text="Index", row=1, col=1)
        self._figs["performance"].update_yaxes(title_text="Trend", row=1, col=1)
        self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X_ssa[1, :], mode = "lines", marker_color = "green"), row=2, col=1)
        self._figs["performance"].update_xaxes(title_text="Index", row=2, col=1)
        self._figs["performance"].update_yaxes(title_text="Seasonality", row=2, col=1)

        
        self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X_ssa[2, :], mode = "lines", marker_color = "red", name = "Residuals"), row=3, col=1)
        self._figs["performance"].update_xaxes(title_text="Index", row=3, col=1)
        self._figs["performance"].update_yaxes(title_text="Residuals", row=3, col=1)
        self._figs["performance"].update_layout(title = "SSA Decomposition", showlegend = False)        
        #pd.DataFrame(res_std, columns=['Residuals']) > np.abs(std_errors)
        self._figs["outliers"] = make_subplots(rows=1, cols=1)
        self._figs["outliers"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X.loc[:, 'y'], mode = "lines", marker_color = "black", name = "Data"), row=1, col=1)   
        self._figs["outliers"].add_trace(go.Scatter(x = X.loc[self.anomalies == True, 'ds'], y = X.loc[self.anomalies == True, 'y'], mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row=1, col=1)
        self._figs["outliers"].update_layout(title = "Outliers Detection", xaxis_title = "Index", yaxis_title = "Values", showlegend = True)
        
        self.update_predict(X, reset_fig=True, update_fig=False)
        return self
    
    def update_predict(self, X, reset_fig = False, update_fig = True):
        if reset_fig:
            self._figs["predict"] = make_subplots(rows=1, cols=1)
            self._figs["predict"].update_layout(title = "Prediction", xaxis_title = "Index", yaxis_title = "Values", showlegend = True)
        if update_fig:
            self._figs["predict"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X.loc[:, 'y'], mode = "lines", marker_color = "blue", name = "Data"), row=1, col=1)
            self._figs["predict"].add_trace(go.Scatter(x = X.loc[X["anomaly"] == True, 'ds'], y = X.loc[X["anomaly"] == True, 'y'], mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row=1, col=1)


    def predict(self, X):
        n_data = self.data_sc.transform(X.loc[:, ['y']])
        X_ssa = self._model.transform(n_data.transpose())
        X_ssa = np.reshape(X_ssa, (X_ssa.shape[1], X_ssa.shape[2]))
        res_std = self._std_model.transform(np.reshape(X_ssa[2, :], (-1, 1)))
        res_std = pd.DataFrame(res_std, columns=['Residuals'])
        anomaly = np.abs(res_std) > self.threshold
        X['anomaly'] = anomaly
        return X
    
if __name__ == "__main__":

    data = pd.read_excel("./data/time_series_train.xlsx")
    ssa = SSAAnomalyDetection()
    ssa.fit(data)
    
    n_data = pd.read_excel("./data/time_series_test.xlsx")
    y = ssa.predict(n_data)
    ssa.update_predict(y)

    ssa._figs["performance"].show()
    ssa._figs["outliers"].show()
    ssa._figs["predict"].show()

    
    print("Done!")