import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from warnings import warn
import mlflow

from copy import deepcopy

import plotly
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

def plot_model(x, outliers, plot :str = "outliers",  y = None,  name_x : str = None, name_y : str = None):
        if plot == "outliers":    
            assert y is not None, "y must be provided for outliers plot"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x = x.loc[outliers == False], y =y.loc[outliers == False], mode = "markers",
                                    marker_color = "black", name = "Inliners"))
            fig.add_trace(go.Scatter(x = x.loc[outliers == True], y = y.loc[outliers == True], mode = "markers",
                                    marker_symbol= "x", marker_color="red",                                    
                                    name = "Outliers"))
            fig.update_layout(title = "Outliers Detection", xaxis_title = name_x, yaxis_title = name_x)
            return fig
            
        
        elif plot == "train":
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Validation Loss"))
            fig.add_trace(go.Bar(x = x.index, y = x, name = "Loss"))
            fig.add_trace(go.Bar(x = y.index, y = y, name = "Validation Loss"))
            fig.update_layout(title = "Training Loss", xaxis_title = "Epoch", yaxis_title = "Loss")
            return fig
        elif plot =="time_series":

            fig = go.Figure()
            fig.add_trace(go.Scatter(x =x.index.to_series().loc[outliers == False].reset_index(drop = True), y =x.loc[outliers == False], mode = "markers",
                                    marker_color = "black", name = "Inliners"))
            fig.add_trace(go.Scatter(x =x.index.to_series().loc[outliers == True].reset_index(drop = True), y =x.loc[outliers == True], mode = "markers",
                                    marker_symbol= "x", marker_color="red",                                    
                                    name = "Outliers"))
            fig.update_layout(title = "Outliers in Time Series.", xaxis_title = "Index", yaxis_title = name_x)
            return fig

@keras.saving.register_keras_serializable(package="MMLW")
class AnomalyDetector(tf.keras.models.Model):
  def __init__(self, neuron_no_enc : list, neuron_no_dec : list, act_enc :str, act_dec:str, layer_no:int, window:int, *args, **kwargs):
    super(AnomalyDetector, self).__init__()
    self.layer_no = layer_no
    self.window = window
    self.neuron_no_enc = neuron_no_enc
    self.neuron_no_dec = neuron_no_dec
    self.act_enc = act_enc
    self.act_dec = act_dec

    self.encoder = tf.keras.Sequential()
    self.encoder.add(tf.keras.layers.Input(shape=(window,)))
    for i in range(0, layer_no):
          self.encoder.add(tf.keras.layers.Dense(neuron_no_enc[i], activation=act_enc))

    self.decoder = tf.keras.Sequential()
    for i in range(0, layer_no):
          self.decoder.add(tf.keras.layers.Dense(neuron_no_dec[i], activation=act_dec))
    
    
    self.decoder.add(tf.keras.layers.Dense(window, activation='linear'))
    

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def get_config(self):
        base_config = super().get_config()
        config = dict()
        config["layer_no"] = self.layer_no
        config["window"] = self.window
        config["neuron_no_enc"] = self.neuron_no_enc
        config["neuron_no_dec"] = self.neuron_no_dec
        config["act_enc"] = self.act_enc
        config["act_dec"] = self.act_dec

        return {**base_config, **config}
    
  @classmethod
  def from_config(cls, config):
        return cls(**config)



class Autoencoder(BaseEstimator):
    _model : None
    _figs : dict[str, go.Figure] = dict().copy() 

    def __init__(self,  *, layer_no = 3, window = 10, epoch_no = 10, batch_size = 64, shuffle = False, threshold_for_anomaly = 3, neuron_no_enc, neuron_no_dec, act_enc, act_dec):
            """ Autoencoder wrapper for anomaly detection.
            - layer_no: int, number of layers
            - window: int, window size for sliding window
            - epoch_no: int, number of epochs for training
            - batch_size: int, batch size for training
            - shuffle: bool, shuffle data during training
            - threshold_for_anomaly: float, threshold for anomaly detection
            - neuron_no_enc: list, number of neurons for encoder layers
            - neuron_no_dec: list, number of neurons for decoder layers
            - act_enc: str, activation function for encoder layers
            - act_dec: str, activation function for decoder layers
            """
            self.layer_no = layer_no
            self.window = window
            self.epoch_no = epoch_no
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.threshold_for_anomaly = threshold_for_anomaly
            self.neuron_no_enc = neuron_no_enc
            self.neuron_no_dec = neuron_no_dec
            self.act_enc = act_enc
            self.act_dec = act_dec


    
    def fit(self, X, y=None):
            
            if "y" not in X.columns:
                raise ValueError("Column named 'y' must be provided for prediction (in variable X)") 
            if "ds" not in X.columns:
                raise ValueError("Column named 'ds' must be provided for labeling (in variable X)") 
            self._sc = StandardScaler()
            target_data = self._sc.fit_transform(X.loc[:, ["y"]])
            train_data = train_test_split(target_data, test_size=0.2, shuffle=self.shuffle, random_state=42)[0]

            stride = train_data.strides
            train_data_tf = np.lib.stride_tricks.as_strided(train_data, shape=(len(train_data)-self.window+1, self.window), strides=stride)

            stride = target_data.strides
            target_data_w = np.lib.stride_tricks.as_strided(target_data, shape=(len(target_data)-self.window+1, self.window), strides=stride)

            self._model = AnomalyDetector(self.neuron_no_enc, self.neuron_no_dec, self.act_enc, self.act_dec, self.layer_no, self.window)
            self._model.compile(optimizer='adam', loss='mse')

            reconstructions = self._model.predict(train_data_tf)
            self.train_loss = tf.keras.losses.mae(reconstructions, train_data_tf)

            reconstruction_data = self._model.predict(target_data_w)
            #self.target_loss = tf.keras.losses.mae(reconstruction_data, target_data_w)

 
            self.threshold = np.mean(self.train_loss) + np.std(self.train_loss)*self.threshold_for_anomaly

            rc_d = np.lib.stride_tricks.as_strided(reconstruction_data, shape=target_data.shape, strides=target_data.strides)
            self.target_loss = tf.keras.losses.mae(rc_d, target_data)

            self.anomalies = np.abs(self.target_loss) > self.threshold

            self._history = self._model.fit(train_data_tf, train_data_tf,
                                      epochs=self.epoch_no,
                                      batch_size=self.batch_size,
                                      validation_split=0.1,
                                      shuffle=self.shuffle)
            

            fig = make_subplots(rows=2, cols=2, subplot_titles=("Epoch Loss", "Epoch Validation Loss", "Training MAE Histogram", "Test MAE Histogram"))
            fig.add_trace(go.Bar(x = np.arange(self.epoch_no), y =self._history.history['loss'], name = "Loss"), row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)

            fig.add_trace(go.Bar(x = np.arange(self.epoch_no), y = self._history.history['val_loss'], name = "Validation Loss"), row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=2)

            train_loss = tf.keras.losses.mae(reconstructions, train_data_tf)
            counts, bins = np.histogram(train_loss.numpy(), bins=50)
            bins = 0.5 * (bins[:-1] + bins[1:])
            fig.add_trace(go.Bar(x = bins, y = counts, name = "Training MAE Histogram"), row=2, col=1)
            fig.update_xaxes(title_text="MAE", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
           

            target_loss = tf.keras.losses.mae(reconstruction_data, target_data_w)
            counts, bins = np.histogram(target_loss.numpy(), bins=50)
            bins = 0.5 * (bins[:-1] + bins[1:])
            fig.add_trace(go.Bar(x = bins, y = counts, name = "Test MAE Histogram"), row=2, col=2)
            fig.update_xaxes(title_text="MAE", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=2)
            self._figs["performance"] = fig

            
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x = X.loc[:, "ds"], y = X.loc[:, "y"], mode = "lines", name = "Inliers"))
            #anoms = X.iloc[0:-(self.window-1), :]
            fig.add_trace(go.Scatter(x = X.loc[self.anomalies == True, "ds"], y = X.loc[self.anomalies == True, "y"], mode = "markers",marker_color = "red", marker_symbol = "x", name = "Outliers"))
            
            self._figs["outliers"] = fig  
            self._figs["outliers"].update_xaxes(title_text="Index", row=1, col=1)
            self._figs["outliers"].update_yaxes(title_text="Values", row=1, col=1)


            fig = make_subplots(rows=1, cols=1)
            self._figs["predict"] = fig
            self._figs["predict"].update_xaxes(title_text="Index", row=1, col=1)
            self._figs["predict"].update_yaxes(title_text="Values", row=1, col=1)

            return self
            

    def predict(self, X):

            if "y" not in X.columns:
                raise ValueError("Column named 'y' must be provided for prediction")    

            target_data = self._sc.transform(X.loc[:, ["y"]])
            stride = target_data.strides
            target_data_w = np.lib.stride_tricks.as_strided(target_data,
                                                            shape=(len(target_data) - self.window + 1, self.window),
                                                            strides=stride)

            reconstruction_data = self._model.predict(target_data_w)
            y = np.lib.stride_tricks.as_strided(reconstruction_data, shape=target_data.shape, strides=target_data.strides)
            target_loss = tf.keras.losses.mae(y, target_data)

            anomaly = np.abs(target_loss) > self.threshold

            #anomaly = np.concatenate((anomaly, np.ones((self.window - 1,)) * anomaly[-1]))

            X["anomaly"] = anomaly
            X["y_pred"] = self._sc.inverse_transform(y)

            return X
    
    def update_predict(self, X, reset_fig = False, update_fig = True):
        if reset_fig:
            self._figs["predict"] = make_subplots(rows=1, cols=1)
            self._figs["predict"].update_xaxes(title_text="Index", row=1, col=1)
            self._figs["predict"].update_yaxes(title_text="Values", row=1, col=1)

        if update_fig:
            self._figs["predict"].add_trace(go.Scatter(x = X.loc[:, "ds"], y = X.loc[:, "y"], mode = "lines", name = "data"))
            self._figs["predict"].add_trace(go.Scatter(x = X.loc[:, "ds"], y = X.loc[:, "y_pred"], mode = "lines", name = "prediction"))
            self._figs["predict"].add_trace(go.Scatter(x = X.loc[X["anomaly"] > 0, "ds"], y = X.loc[X["anomaly"] > 0, "y"], mode = "markers",marker_color = "red", marker_symbol = "x", name = "Outliers"))

         
        #def score(self, X, y=None):
        #    return np.mean(self.anomalies)


if __name__ == "__main__":
     
    data = pd.read_excel("./data/time_series_train.xlsx")
    ae = Autoencoder(layer_no =6, window =250, epoch_no = 10, batch_size = 64, shuffle = False, threshold_for_anomaly = 3, neuron_no_enc = [30, 25, 20, 15, 10,5], neuron_no_dec = [5,10, 15, 20, 25, 30], act_enc = 'relu', act_dec = 'relu')
    ae.fit(data)
    n_data = pd.read_excel("./data/time_series_test_0.xlsx")
    y = ae.predict(n_data)
    ae.update_predict(y)

    ae._figs["performance"].show() 
    ae._figs["outliers"].show()
    ae._figs["predict"].show()

    print("Done!")