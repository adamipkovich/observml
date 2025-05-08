import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
import plotly
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import logging 
class ExponentialSmoothingAnomaly(BaseEstimator):

    _model = None
    _sc = None
    anomalies = None
    _figs : dict[str, go.Figure] = dict().copy()

    def __init__(self, *, trend : str, seasonal : str, seasonal_periods : int, threshold_for_anomaly : float, freq : str):
        """ - trend : str (default = None)
                Type of trend component.
            -damped_trend : bool (default = False)
                Should the trend component be damped.
            -seasonal : str (default = None)
                Type of seasonal component.
            -seasonal_periods : int (default = None)
                The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily data with a weekly cycle.
            -freq : str (default = None)
                Frequency of the time series. A Pandas offset or 'B', 'D', 'W', 'M', 'A', or 'Q'.
            -threshold_for_anomaly : float (default = 3)
                Threshold for anomaly detection."""
        
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.threshold_for_anomaly = threshold_for_anomaly
        self.freq = freq
        self._sc = StandardScaler()
        
    def fit(self, X, y=None):
        if 'ds' not in X.columns:
            raise ValueError("The input data should have a column named 'ds' for the date and time.")
        if 'y' not in X.columns:
            raise ValueError("The input data should have a column named 'y' for the target variable.")
        


        X.set_index('ds', drop=True, inplace=True)
        sc_data = self._sc.fit_transform(X)
        nX =pd.Series(sc_data.flatten(), index=X.index) 
        self._model = ExponentialSmoothing(X, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods) #freq=self.freq
        #X, 
        self._model = self._model.fit(optimized=True)
        forecasts = self._model.fittedvalues

        errors = np.abs(nX - forecasts)

        y = forecasts.to_frame() # self._sc.inverse_transform(forecasts.to_frame())
        
        counts, bins = np.histogram(errors, bins=50)
        bins = 0.5 * (bins[:-1] + bins[1:])
        self._figs["performance"] = make_subplots(rows=2, cols=1)

        self._figs["performance"].add_trace(go.Scatter(x = X.loc[:, 'y'], y = y[0], name = "Standardized Error", mode = "markers"), row=1, col=1)
        self._figs["performance"].update_xaxes(title_text="Input data", row=1, col=1)
        self._figs["performance"].update_yaxes(title_text="Forecast data", row=1, col=1)
        self._figs["performance"].add_trace(go.Bar(x = bins, y = counts, name = "Forecast error histogram"), row=2, col=1)
        self._figs["performance"].update_xaxes(title_text="Absolute Error", row=2, col=1)
        self._figs["performance"].update_yaxes(title_text="Count", row=2, col=1)

        self.threshold = np.mean(errors) + np.std(errors)*self.threshold_for_anomaly
        self.anomalies = np.abs(errors)>self.threshold

        self._figs["outliers"] = make_subplots(rows=1, cols=1)

        ind = X.index
        self._figs["outliers"].add_trace(go.Scatter(x = ind, y = X.loc[:, 'y'], mode = "lines", marker_color = "black", name = "Data"), row=1, col=1)   
        self._figs["outliers"].add_trace(go.Scatter(x = ind, y = y[0], mode = "lines", marker_color = "blue", name = "Prediction"), row=1, col=1)   
        self._figs["outliers"].add_trace(go.Scatter(x = ind[self.anomalies == True], y = X.loc[self.anomalies == True, "y"], mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row=1, col=1)
        self._figs["outliers"].update_layout(title = "Outliers Detection", xaxis_title = "Index", yaxis_title = "Values")

        self._figs["predict"] = make_subplots(rows=1, cols=1)
        self._figs["predict"].update_xaxes(title_text="Index", row=1, col=1)
        self._figs["predict"].update_yaxes(title_text="Values", row=1, col=1)


        self._is_fitted_ = True
        return self
    

    def predict(self, X):
        if not self._is_fitted_:
            raise ValueError("The model is not fitted yet.")
        
        if 'ds' not in X.columns:
            logging.warning("The input data should have a column named 'ds' for the date and time.")
            #check if the index is datetime
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("The input data should have a column named 'ds' for the date and time.")
        else:
            X.set_index('ds', drop=True, inplace=True)

        sc_data = self._sc.transform(X)
        nX =pd.Series(sc_data.flatten(), index=X.index)
        forecast = self._model.predict(start=nX.index[0], end=nX.index[-1])
        y = self._sc.inverse_transform(forecast.to_frame())

        errors = np.abs(nX - forecast)
        anomalies = errors>self.threshold
        X["forecast"] = y 
        X['anomalies'] = anomalies  
        return X
    
    def update_predict(self, X, reset_fig = False, update_fig = True):
        if not self._is_fitted_:
            raise ValueError("The model is not fitted yet.")
        
        if reset_fig:
            self._figs["predict"] = make_subplots(rows=1, cols=1)
            self._figs["predict"].update_xaxes(title_text="Index", row=1, col=1)
            self._figs["predict"].update_yaxes(title_text="Values", row=1, col=1)
        if update_fig:
            self._figs["predict"].add_trace(go.Scatter(x = X.index, y = X.loc[:, 'y'], mode = "lines", marker_color = "black", name = "Data"), row=1, col=1)
            self._figs["predict"].add_trace(go.Scatter(x = X.index, y = X.loc[:, 'forecast'], mode = "lines", marker_color = "blue", name = "Prediction"), row=1, col=1)
            self._figs["predict"].add_trace(go.Scatter(x = X.index[X["anomalies"] == True], y = X.loc[X["anomalies"] == True, 'y'], mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row=1, col=1)
            self._figs["predict"].update_layout(title = "Prediction", xaxis_title = "Index", yaxis_title = "Values", showlegend = False)
   


if __name__ == "__main__":
    data = pd.read_excel("./data/time_series_train.xlsx")
    es = ExponentialSmoothingAnomaly(trend="mul", seasonal=None, seasonal_periods=None, threshold_for_anomaly=2.5, freq=None)    
    es.fit(data)

    n_data = pd.read_excel("./data/time_series_test.xlsx")
    es.predict(n_data)
    es.update_predict(n_data)

    es._figs["performance"].show()
    es._figs["outliers"].show()
    es._figs["predict"].show()

    print("Done!")