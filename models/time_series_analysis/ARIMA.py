import numpy as np
import pandas as pd

from pmdarima.arima import auto_arima, decompose

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
import plotly
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import mlflow

class ARIMAAnomalyDetector(BaseEstimator):
    _model = None
    _sc = None
    anomalies = None
    _figs : dict[str, go.Figure] = dict().copy()
    
    def __init__(self, *, start_p : int, d : int, start_q : int, max_p : int, max_q : int, seasonal : bool, threshold_for_anomaly : float):
        """  
             - start_p: start value for p
             - d: differencing order
             - start_q: start value for q
             - max_p: max value for p
             - max_q: max value for q
             - seasonal: True/False
             - threshold_for_anomaly: threshold for anomaly detection"""
        
        
        self.threshold_for_anomaly = threshold_for_anomaly
        self.start_p = start_p
        self.d = d
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self._sc = StandardScaler()

    def fit(self, X, y=None):
        if "y" not in X.columns:
                raise ValueError("Column named 'y' must be provided for prediction (in variable X)") 
        if "ds" not in X.columns:
                raise ValueError("Column named 'ds' must be provided for labeling (in variable X)") 
        #target_data = self._sc.fit_transform(X.loc[:, ["y"]])
        
        #mlflow.autolog(disable=True)

        self._model = auto_arima(X.loc[:, ["y"]], start_p=self.start_p, d=self.d, start_q=self.start_q, max_p=self.max_p, max_q=self.max_q, seasonal=self.seasonal, n_jobs=-1, stepwise=False)
        #forecasts = self._model.arima_res_.forecasts
        y = self._model.arima_res_.forecasts #self._sc.inverse_transform(self._model.arima_res_.forecasts)
        errors = np.abs( X.loc[:, 'y'] - y.flatten())

        #self.decomposed = decompose(X.loc[:, 'y'], type_ = "additive", m = self.m)

        counts, bins = np.histogram(errors, bins=50)
        bins = 0.5 * (bins[:-1] + bins[1:])
        self._figs["performance"] = make_subplots(rows=2, cols=1)

        self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'y'], y = y.flatten(), name = "Standardized Error", mode = "markers"), row=1, col=1)
        self._figs["performance"].update_xaxes(title_text="Input data", row=1, col=1)
        self._figs["performance"].update_yaxes(title_text="Forecast data", row=1, col=1)
        self._figs["performance"].add_trace(go.Bar(x = bins, y = counts, name = "Forecast error histogram"), row=2, col=1)
        self._figs["performance"].update_xaxes(title_text="Absolute Error", row=2, col=1)
        self._figs["performance"].update_yaxes(title_text="Count", row=2, col=1)
        
        #self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = self.decomposed.seasonal, mode = "lines", marker_color = "blue", name = "Seasonal"), row=1, col=1)
        #self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = self.decomposed.trend, mode = "lines", marker_color = "green", name = "Trend"), row=2, col=1)
        #self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = self.decomposed.random, mode = "lines", marker_color = "red", name = "Residual"), row=3, col=1)
        #self._figs["performance"].update_layout(title = "Decomposition", xaxis_title = "Index", yaxis_title = "Values")
        
        self.threshold = np.mean(errors) + np.std(errors)*self.threshold_for_anomaly
        self.anomalies = np.abs(errors)>self.threshold
        

        self._figs["outliers"] = make_subplots(rows=1, cols=1)
        self._figs["outliers"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X.loc[:, 'y'], mode = "lines", marker_color = "black", name = "Data"), row=1, col=1)   
        self._figs["outliers"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = y.flatten(), mode = "lines", marker_color = "blue", name = "Prediction"), row=1, col=1)   
        self._figs["outliers"].add_trace(go.Scatter(x = X.loc[self.anomalies == True, 'ds'], y = X.loc[self.anomalies == True, 'y'], mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row=1, col=1)
        self._figs["outliers"].update_layout(title = "Outliers Detection", xaxis_title = "Index", yaxis_title = "Values")
        
        self._figs["predict"] = make_subplots(rows=1, cols=1)
        self._figs["predict"].update_xaxes(title_text="Index", row=1, col=1)
        self._figs["predict"].update_yaxes(title_text="Values", row=1, col=1)


        #mlflow.autolog(disable=False)
        return self

    def predict(self, X):
        #target = self._sc.transform(X.loc[:, ["y"]])
        forecast = self._model.predict(n_periods = X.shape[0])
        y = forecast.reset_index(drop=True)
        errors = np.abs( X.loc[:, 'y'] - y)
        #self.threshold = np.mean(errors) + np.std(errors)*self.threshold_for_anomaly
        anomalies = errors>self.threshold
        X["anomaly"] = anomalies
        X["y_pred"] = y  
        return X
    
    def update_predict(self, X, reset_fig = False, update_fig = True):
        if reset_fig:
            self._figs["predict"] = make_subplots(rows=1, cols=1)
            self._figs["predict"].update_xaxes(title_text="Index", row=1, col=1)
            self._figs["predict"].update_yaxes(title_text="Values", row=1, col=1)
            
        if update_fig:
            self._figs["predict"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X.loc[:, 'y'], mode = "lines", marker_color = "black", name = "Data"), row=1, col=1)   
            self._figs["predict"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X["y_pred"], mode = "lines", marker_color = "blue", name = "Prediction"), row=1, col=1)   
            self._figs["predict"].add_trace(go.Scatter(x = X.loc[X["anomaly"] == True, 'ds'], y = X.loc[X["anomaly"] == True, 'y'], mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row=1, col=1)
        
if __name__ == "__main__":
    data = pd.read_excel("./data/time_series_train.xlsx")
    arima = ARIMAAnomalyDetector(start_p=10, d=None, start_q=10, max_p=200, max_q=200, seasonal=False, threshold_for_anomaly=3)
    arima.fit(data)
    n_data = pd.read_excel("./data/time_series_test.xlsx")
    arima.predict(n_data)

    arima.update_predict(n_data)
    arima._figs["performance"].show()
    arima._figs["outliers"].show()
    arima._figs["predict"].show()

    print("Done!")