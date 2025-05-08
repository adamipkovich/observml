# Description: DBSCAN Anomaly Detection Model
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score,  adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN
import mlflow


# http://www.sefidian.com/2022/12/18/how-to-determine-epsilon-and-minpts-parameters-of-dbscan-clustering/
### select minsample -> then use elbow k nearest neighbors to select epsilon



class DBSCANAnomalyDetection(BaseEstimator):
    _model : DBSCAN = None
    _std_scaler : StandardScaler = None
    _figs : dict[str, go.Figure] = {}   


    def __init__(self,  *, eps=2, min_samples= None):
        self.eps = eps
        self.min_samples = min_samples
        self._std_scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.scheme = X.columns
        if self.min_samples is None:
            self.min_samples = 2 * len(self.scheme)
        
        sc_X = self._std_scaler.fit_transform(X)

        
        #neighbors = NearestNeighbors(n_neighbors=min_samples)
        #neighbors_fit = neighbors.fit(data)
        #distances, indices = neighbors_fit.kneighbors(data)
        #distances = np.sort(distances, axis=0)
        #distances = distances[:,-1] ## moving abverage of distances' distances

        #import matplotlib.pyplot as plt
        #plt.plot(distances)
        #plt.show()

        #mad = np.convolve(distances, np.ones(min_samples)/min_samples, mode='valid') #/np.arange(1, len(distances)+1)
        #mad[0:-2] - mad[1:-1]


        self.eps = 2 # mad[(mad[0:-2] - mad[1:-1]).argmax() + 1] ## (mad[0:-2] - mad[1:-1]).argmax()
        self._model = DBSCAN(eps = self.eps, 
                             min_samples = self.min_samples).fit(sc_X)
        self.anomalies = self._model.labels_ == -1

        ## TODO: ADD a figure for each column for i in self.scheme -> outliers_{i} = make_subplots(rows=1, cols=1, subplot_titles=("Outliers",))
 

        for i, col in enumerate(self.scheme):
            self._figs[f"{col}_performance"] = make_subplots(rows=1, cols=1, subplot_titles=(f"{col} Performance",))
            counts, bins = np.histogram(X.loc[self.anomalies, col], bins=50)
            bins = 0.5 * (bins[:-1] + bins[1:])
            self._figs[f'{col}_performance'].add_trace(go.Bar(x = bins, y = counts, name = "Anomaly Distributions."), row=1, col=1)
            self._figs[f'{col}_performance'].update_xaxes(title_text=col, row=1, col=1)
            self._figs[f'{col}_performance'].update_yaxes(title_text='Anomaly count', row=1, col=1)
            self._figs[f'{col}_performance'].update_layout(showlegend=False)
            #self._figs['performance_{col}'].update_layout(title_text='DBSCAN Anomaly Detection', showlegend=False, width=1000, height=200*X.shape[1])

        
        for i, col in enumerate(self.scheme):
            self._figs[f"{col}_outliers"] = make_subplots(rows=1, cols=1, subplot_titles=("Outliers",))
            self._figs[f"{col}_outliers"].add_trace(go.Scatter(x=X.index, y=X.loc[:, col], mode='lines', name=col, line=dict(color='blue')), row=1, col=1)
            self._figs[f"{col}_outliers"].add_trace(go.Scatter(x=X.index[self.anomalies], y=X.iloc[self.anomalies, i], mode='markers', name='Outliers', marker=dict(color='red')), row=1, col=1)
            self._figs[f"{col}_outliers"].update_xaxes(title_text='Index', row=1, col=1)
            self._figs[f"{col}_outliers"].update_yaxes(title_text=col, row=1, col=1)
            self._figs[f"{col}_outliers"].update_layout(showlegend=False, title_text=f'DBSCAN Outliers for {col}')
        
        #self._figs[f"outliers_{col}"].update_layout(title_text=f'DBSCAN Outliers for {col}', width=1000, height=200*X.shape[1], showlegend=False)
        self.update_predict(X, reset_fig = True, update_fig = False)

        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        sc_x = self._std_scaler.transform(X)
        nX = X.copy()
        nX["outliers"] = self._model.fit_predict(sc_x) == -1
        return nX
    
    def update_predict(self, X, reset_fig = False, update_fig = True):
        if reset_fig:
            for i, col in enumerate(self.scheme):
                self._figs[f"{col}_predict"] = make_subplots(rows=1, cols=1, subplot_titles=("Prediction",))
                self._figs[f"{col}_predict"].update_xaxes(title_text='Index', row=1, col=1)
                self._figs[f"{col}_predict"].update_yaxes(title_text=col, row=1, col=1)
                self._figs[f"{col}_predict"].update_layout(title_text=f'Outlier prediction for {col}', showlegend=False)

        if update_fig:
            for i, col in enumerate(self.scheme):
                self._figs[f"{col}_predict"].add_trace(go.Scatter(x=X.index, y=X.loc[:, col], mode='lines', name=col, line=dict(color='blue')), row=1, col=1)
                self._figs[f"{col}_predict"].add_trace(go.Scatter(x=X.index[X["outliers"]], y=X.loc[X["outliers"], col], mode='markers', name='Outliers', marker=dict(color='red')), row=1, col=1)
            
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    data = pd.read_excel("./data/pump_train.xlsx")

    dbscan = DBSCANAnomalyDetection()
    dbscan.fit(data)
    n_data = pd.read_excel("./data/pump_test.xlsx")
    y = dbscan.predict(n_data)
    dbscan.update_predict(y)

    #fig = plot_model(n_data.loc[:, "sensor_01"], y["outliers"], y = n_data.loc[:, "sensor_02"], name_x = "sensor_01", name_y = "sensor_02", plot = "outliers")
    #fig = plot_model(n_data.loc[:, "sensor_01"], y["outliers"], y = n_data.loc[:, "sensor_02"], name_x = "sensor_01", name_y = "sensor_02", plot = "time_series")

    print("Done!")