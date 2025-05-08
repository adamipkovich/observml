import pandas as pd
from prophet import Prophet
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error

from copy import deepcopy

import plotly
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

class ProphetAnomalyDetection(BaseEstimator):
    _model : any = None
    _future : pd.DataFrame = None
    _figs : dict[str, go.Figure] = dict().copy() 


    def __init__(self, * , periods = 0, factor = 1.0, forecast_window = 200):
            """ This model is a wrapper for the Prophet model, to be able to detect time-series outliers.
             It incorporates mlflow for logging and tracking purposes
            :param data: pandas dataframe with columns ds and y
            :param params: a dict with the following keys:
            - factor (float): the factor for the anomaly detection. Default is 1.0
            """
            
            self.periods = periods
            self.factor = factor
            self.forecast_window = forecast_window
            self._model = Prophet() ## build prophet model
            self._sc = StandardScaler()


            

    def fit(self, X, y=None):
        
        assert 'ds' in X.columns, "ds column must be present in the data"
        assert 'y' in X.columns, "y column must be present in the data"
        
        self._model.fit(X) ## fit hte model on the supplied data
        future = self._model.make_future_dataframe(periods=self.periods) ## create a future dataframe
        forecast = self._model.predict(future)    
        forecast_df = forecast[['ds','yhat','yhat_upper','yhat_lower']] ## get the forecasted values
        
        forecasting_final = pd.merge(forecast_df, X, how='inner', left_on='ds', right_on='ds')

        forecasting_final['error'] = forecasting_final['y'] - forecasting_final['yhat'] ## calculate the error
        forecasting_final['uncertainty'] = forecasting_final['yhat_upper'] - forecasting_final['yhat_lower'] ## calculate the uncertainty
        forecasting_final['anomaly']  = (
                  forecasting_final.apply(lambda x: 'Yes'
                  if(np.abs(x['error']) >  self.factor*x['uncertainty']) else 'No', axis = 1))

        self.anomalies = forecasting_final['anomaly']


        self._figs["performance"]  = make_subplots(rows=1, cols=2, subplot_titles=("Error Distribution", "Anomaly Amount in Training"))
        self._figs["performance"].add_trace(go.Histogram(x = forecasting_final['error'], name = "Error Distribution"), row = 1, col = 1)
        self._figs["performance"].update_layout(xaxis_title = "Error Value", yaxis_title = "Count")
        
        self._figs["performance"].add_trace(go.Bar(x = forecasting_final['anomaly'].value_counts().index, y = forecasting_final['anomaly'].value_counts().values, name = "Anomaly Amount"), row = 1, col = 2)
        self._figs["performance"].update_layout(xaxis_title = "Is Anomaly", yaxis_title = "Anomaly count" )

        self._figs["outliers"] =make_subplots(rows=1, cols=1)
        self._figs["outliers"].add_trace(go.Scatter(x = forecasting_final['ds'], y = forecasting_final['y'], mode = "lines", name = "Actual"), row = 1, col = 1)
        self._figs["outliers"].add_trace(go.Scatter(x = forecasting_final['ds'], y = forecasting_final['yhat'], mode = "lines", name = "Forecasted"), row = 1, col = 1)
        self._figs["outliers"].add_trace(go.Scatter(x = forecasting_final['ds'], y = forecasting_final['yhat_lower'], mode = "lines", name = "Lower confidence",marker_color = "red"), row = 1, col = 1)
        self._figs["outliers"].add_trace(go.Scatter(x = forecasting_final['ds'], y = forecasting_final['yhat_upper'], mode = "lines", name = "Upper confidence",marker_color = "blue"), row = 1, col = 1)
        
        self._figs["outliers"].add_trace(go.Scatter(x = forecasting_final['ds'].loc[forecasting_final['anomaly'] == 'Yes'], y = forecasting_final['y'].loc[forecasting_final['anomaly'] == 'Yes'], 
                                                    mode = "markers", 
                                                    marker_color = "red",
                                                    marker_symbol  = "x",
                                                    name = "Outliers"))


        self.update_predict(X, reset_fig = True, update_fig = False)
        
        return self


    def predict(self, X):

            assert 'ds' in X.columns, "ds column must be present in the data"
            assert 'y' in X.columns, "y column must be present in the data"

            forecast = self._model.predict(X)
            forecast_df = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
            forecasting_final = pd.merge(forecast_df, X, how='inner', left_on='ds', right_on='ds')
            forecasting_final['error'] = forecasting_final['y'] - forecasting_final['yhat']
            forecasting_final['uncertainty'] = forecasting_final['yhat_upper'] - forecasting_final['yhat_lower']
            forecasting_final['anomaly'] = (
                  forecasting_final.apply(lambda x: 'Yes'
                  if(np.abs(x['error']) >  self.factor*x['uncertainty']) else 'No', axis = 1))

            ## forecast the length of incoming data...
            forecasting_final.rename(columns = {'yhat': 'y_pred'}, inplace = True)
            #future = self._model.make_future_dataframe(periods=10, freq = "MS")
            dates = pd.DataFrame([])
            dates["ds"] = pd.date_range(X["ds"].iloc[-1], periods = X.shape[0], freq= pd.Timedelta(X["ds"].iloc[-1] - X["ds"].iloc[-2]))
            self.forecast = self._model.predict(dates)
            self.forecast = self.forecast.loc[:, ["ds", "yhat", "yhat_lower", "yhat_upper"]]
            return forecasting_final #, anomaly_ratio, MAE, MSE, RMSE, MAPE, mean_uncertainty

    def update_predict(self, X, reset_fig = False, update_fig = True):
        if reset_fig:
            self._figs["predict"] = make_subplots(rows=1, cols=1)
            self._figs["predict"].update_layout(title = "Prediction", xaxis_title = "Index", yaxis_title = "Values")
            
        if update_fig:
            self._figs["predict"].add_trace(go.Scatter(x = X['ds'], y = X['y'], mode = "lines", name = "Actual", marker_color = "black"), row = 1, col = 1)
            self._figs["predict"].add_trace(go.Scatter(x = X['ds'], y = X['y_pred'], mode = "lines", name = "Predicted", marker_color = "orange"), row = 1, col = 1)
            self._figs["predict"].add_trace(go.Scatter(x = X['ds'], y = X['yhat_lower'], mode = "lines", name = "Lower confidence",marker_color = "red"), row = 1, col = 1)
            self._figs["predict"].add_trace(go.Scatter(x = X['ds'], y = X['yhat_upper'], mode = "lines", name = "Upper confidence",marker_color = "blue"), row = 1, col = 1)
            self._figs["predict"].add_trace(go.Scatter(x = X['ds'].loc[X['anomaly'] == 'Yes'], y = X['y'].loc[X['anomaly'] == 'Yes'],  mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row = 1, col = 1)
            
            ## Add forecast update

            self._figs["predict"].add_trace(go.Scatter(x = self.forecast['ds'], y = self.forecast['yhat'], mode = "lines", name = "Forecast Prediction", marker_color = "orange", line=dict(dash='dash')), row = 1, col = 1)
            self._figs["predict"].add_trace(go.Scatter(x = self.forecast['ds'], y = self.forecast['yhat_lower'], mode = "lines", name = "Forecast Lower confidence",marker_color = "red", line=dict(dash='dash')), row = 1, col = 1)
            self._figs["predict"].add_trace(go.Scatter(x = self.forecast['ds'], y = self.forecast['yhat_upper'], mode = "lines", name = "Forecast Upper confidence",marker_color = "blue", line=dict(dash='dash')), row = 1, col = 1)
            self._figs["predict"].update_layout(title = "Forecasted Values", xaxis_title = "Date", yaxis_title = "Value")

if __name__ == "__main__":
     
    from pycaret.time_series import *
    data = pd.read_excel("./data/time_series_train.xlsx")
    ph = ProphetAnomalyDetection()
    ph.fit(data)
    n_data = pd.read_excel("./data/time_series_test.xlsx")
    y = ph.predict(n_data)
    ph.update_predict(y)
    ph._figs["predict"].show()
    print("Done!")