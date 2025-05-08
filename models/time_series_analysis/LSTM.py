import pandas as pd
import numpy as np
import tensorflow as tf
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

from tensorflow.keras.models import Sequential, save_model, load_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import timeseries_dataset_from_array

def sequential_model(layer_no : int = 1, cell_no : int = 1, seq_length : int = 1, loss_func =MeanSquaredError(),  metrics_func = [tf.keras.metrics.R2Score()] ):
          model = Sequential()
          model.add(tf.keras.layers.InputLayer((seq_length, 1)))
          for _ in range(0,layer_no):
                model.add(tf.keras.layers.LSTM(cell_no, return_sequences=False))
          model.add(tf.keras.layers.Dense(units=1))
          model.compile(loss=loss_func, optimizer=tf.keras.optimizers.Adam(), metrics=metrics_func)
          return model


class LSTM(BaseEstimator):
      
    _model : Sequential = None
    _future : pd.DataFrame = None
    _anomaly = None
    _figs : dict[str, go.Figure] = dict()
    y_pred = None  
    def __init__(self, *, seq_length : int = 1, layer_no : int = 1, cell_no : int = 6, epoch_no : int = 50, batch_size : int = 32, shuffle : bool = False, patience : int = 10, threshold_for_anomaly : float = 3):
            """Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.
            - seq_length : int
                Length of the sequence.
                - target_var : str
                Name of the target variable.
                - layer_no : int (default = 1)
                Number of LSTM layers.
                - cell_no : int (default = 1)
                Number of cells in LSTM layer.
                - epoch_no : int (default = 100)
                Number of epochs.
                - batch_size : int (default = 32)
                Batch size.
                - shuffle : bool (default = True)
                Shuffle data.
                - patience : int (default = 10)
                Number of epochs with no improvement after which training will be stopped.
                - threshold_for_anomaly : float (default = 3)
                Threshold for anomaly detection."""
            
             #scaling data
            self.seq_length = seq_length
            self.layer_no = layer_no
            self.cell_no = cell_no
            self.epoch_no = epoch_no
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.patience = patience
            self.threshold_for_anomaly = threshold_for_anomaly
            self.data_sc = StandardScaler()
            self.error_sc = StandardScaler()
            self._figs  = dict()
            
    def fit(self, X, y=None):
            
            target_data = X["y"].values
            target_data = self.data_sc.fit_transform(target_data.reshape(-1,1))
            input_data = target_data[:-(self.seq_length)]
            targets = target_data[self.seq_length:]

            dataset_train = timeseries_dataset_from_array(input_data, targets, sequence_length=self.seq_length,
                                                    batch_size=1, shuffle=self.shuffle,
                                                    start_index=0, end_index=int(len(input_data)*0.8))

            dataset_val = timeseries_dataset_from_array(input_data, targets, sequence_length=self.seq_length,
                                                          batch_size=1, shuffle=self.shuffle,
                                                          start_index=int(len(input_data) * 0.8), end_index=None)
            
            X_train = []
            y_train = []
            for input, target in dataset_train:
                  X_train.append(input)
                  y_train.append(target)
            X_train = np.array(X_train)
            X_train = X_train.reshape(len(X_train), X_train.shape[2])
            y_train = np.array(y_train)
            y_train = y_train.reshape(len(y_train),1)

            X_val = []
            y_val = []
            for input, target in dataset_val:
                  X_val.append(input)
                  y_val.append(target)
            X_val = np.array(X_val)
            X_val = X_val.reshape(len(X_val), X_val.shape[2])
            y_val = np.array(y_val)
            y_val = y_val.reshape(len(y_val), 1)

            self._model = sequential_model(layer_no = self.layer_no, cell_no = self.cell_no, seq_length = self.seq_length)

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=self.patience,
                                                              mode='min')
            self._history = self._model.fit(X_train, y_train, batch_size=self.batch_size,
                                      epochs=self.epoch_no,
                                      validation_data=(X_val, y_val),callbacks=[early_stopping])
            
            self._figs["performance"] = make_subplots(rows=2, cols=1)
            self._figs["performance"].add_trace(go.Scatter(x=self._history.epoch,
                                                           y=self._history.history['loss'],
                                                            mode='lines',
                                                            name='Train loss'), row=1, col=1)
            self._figs["performance"].add_trace(go.Scatter(x=self._history.epoch, y=self._history.history['val_loss'], mode='lines', name='Validation Loss'), row=1, col=1)    


            self._figs["performance"].update_xaxes(title_text="Index", row=1, col=1)
            self._figs["performance"].update_yaxes(title_text="Loss", row=1, col=1)

            
            #self._figs["performance"].update_layout(title = "Loss Functions", xaxis_title = "Index", yaxis_title = "Loss")
            self._figs["performance"].add_trace(go.Scatter(x=self._history.epoch, y=self._history.history['r2_score'], mode='lines', name='Train'), row=2, col=1)
            self._figs["performance"].add_trace(go.Scatter (x=self._history.epoch, y=self._history.history['val_r2_score'], mode='lines', name='Validation'), row=2, col=1)
            self._figs["performance"].update_xaxes(title_text="Index", row=2, col=1)
            self._figs["performance"].update_yaxes(title_text="R2 Score", row=2, col=1)
            #self._figs["performance"].show()

            prediction_val = self._model.predict(target_data)
            self.y_pred =pd.Series(self.data_sc.inverse_transform(prediction_val).flatten()) 


            errors = prediction_val - target_data #y_val


            std_errors = self.error_sc.fit_transform(errors)
            anomaly = np.abs(std_errors) > self.threshold_for_anomaly


            X['anomaly'] = anomaly
            # plt.figure()
            # sns.scatterplot(df, x=df.index.values, y='y', hue='anomaly')
            # plt.savefig(os.path.join(params.get("output_path"),  "anomaly.png"))
            self.anomalies = anomaly


            self._figs["outliers"] = make_subplots(rows=1, cols=1)
            self._figs["outliers"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X.loc[:, 'y'], mode = "lines", marker_color = "black", name = "Data"), row=1, col=1)   
            self._figs["outliers"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = self.y_pred, mode = "lines", marker_color = "blue", name = "Predicted"), row=1, col=1)           
            self._figs["outliers"].add_trace(go.Scatter(x = X.loc[X["anomaly"] == True, 'ds'], y = X.loc[X["anomaly"] == True, 'y'], mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row=1, col=1)

            self._figs["outliers"].update_layout(title = "Outliers Detection", xaxis_title = "Index", yaxis_title = "Values")
            #self._figs["outliers"].show()
            
            self._figs["predict"] = make_subplots(rows=1, cols=1)
            self._figs["predict"].update_layout(title = "Prediction", xaxis_title = "Index", yaxis_title = "Values")

            return self

    def predict(self, X):

            target_data = X["y"].values
            target_data = self.data_sc.transform(target_data.reshape(-1, 1))
            prediction = self._model.predict(target_data)

            errors = (prediction - target_data)
            std_errors = self.error_sc.transform(errors.reshape(-1, 1))

            anomaly = np.abs(std_errors) > self.threshold_for_anomaly
            X['anomaly'] = anomaly
            X['y_pred'] = self.data_sc.inverse_transform(prediction) 

            ## TODO: return prediction as well...
            return X
    
    def update_predict(self, X, reset_fig = False, update_fig = True):
            
            if reset_fig:
                  self._figs["predict"] = make_subplots(rows=1, cols=1)
                  self._figs["predict"].update_layout(title = "Prediction", xaxis_title = "Index", yaxis_title = "Values")

            if update_fig:
                  self._figs["predict"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X['y'], mode = "lines", marker_color = "black", name = "Data"), row=1, col=1)
                  self._figs["predict"].add_trace(go.Scatter(x = X.loc[:, 'ds'], y = X['y_pred'], mode = "lines", marker_color = "blue", name = "Predicted"), row=1, col=1)
                  self._figs["predict"].add_trace(go.Scatter(x = X.loc[X["anomaly"] == True, 'ds'], y = X.loc[X["anomaly"] == True, 'y'], mode = "markers", marker_color = "red", marker_symbol = "x", name = "Outliers"), row=1, col=1)
            #self._figs["predict"].show()

            
    
 
if __name__ == "__main__":

    data = pd.read_excel("./data/time_series_train.xlsx")
    lstm = LSTM()
    lstm.fit(data)
    
    n_data = pd.read_excel("./data/time_series_test_0.xlsx")
    y = lstm.predict(n_data)
    
    print("Done!")