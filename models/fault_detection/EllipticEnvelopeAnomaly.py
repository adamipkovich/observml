from sklearn.covariance import EllipticEnvelope
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator


class EllipticEnvelopeAnomalyDetection(BaseEstimator):
    _model : EllipticEnvelope = None
    _data_sc : StandardScaler = None
    _figs : dict[str, go.Figure] = {}

    def __init__(self,  *, contamination=0.1):
        
        assert contamination > 0, "Contamination must be greater than 0."
        assert contamination < 0.5, "Contamination must be less than 0.5."

        self._std_scaler = StandardScaler()
        self._model = EllipticEnvelope(contamination = contamination)
        
    def fit(self, X, y=None):

        self.scheme = X.columns
        sc_X = self._std_scaler.fit_transform(X)
        self._model = self._model.fit(sc_X)
        self.anomalies = self._model.predict(sc_X) == -1

        ## Generating figures
        

        for i, col in enumerate(self.scheme):
            self._figs[f"{col}_performance"] = make_subplots(rows=1, cols=1, subplot_titles=("Performance",))
            counts, bins = np.histogram(X.loc[self.anomalies, col], bins=50)
            bins = 0.5 * (bins[:-1] + bins[1:])
            self._figs[f'{col}_performance'].add_trace(go.Bar(x = bins, y = counts, name = "Anomaly Distributions."), row=1, col=1)
            self._figs[f'{col}_performance'].update_xaxes(title_text=col, row=1, col=1)
            self._figs[f'{col}_performance'].update_yaxes(title_text='Anomaly count', row=1, col=1)
            self._figs[f"{col}_performance"].update_layout(showlegend=False, )

        
        for i, col in enumerate(self.scheme):
            self._figs[f"{col}_outliers"] = make_subplots(rows=1, cols=1, subplot_titles=("Outliers",))
            self._figs[f"{col}_outliers"].add_trace(go.Scatter(x=X.index, y=X.loc[:, col], mode='lines', name=col, line=dict(color='blue')), row=1, col=1)
            self._figs[f'{col}_outliers'].add_trace(go.Scatter(x=X.index[self.anomalies], y=X.iloc[self.anomalies, i], mode='markers', name='Outliers', marker=dict(color='red')), row=1, col=1)
            self._figs[f'{col}_outliers'].update_xaxes(title_text='Index', row=1, col=1)
            self._figs[f'{col}_outliers'].update_yaxes(title_text=col, row=1, col=1)
            self._figs[f'{col}_outliers'].update_layout(showlegend=False)

        self.update_predict(X, reset_fig = True, update_fig = False)

        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        sc_x = self._std_scaler.transform(X)
        nX = X.copy()
        nX["outliers"] = self._model.predict(sc_x) == -1
        return nX

    def update_predict(self, X, reset_fig = False, update_fig = True):
        if reset_fig:
            
            for i, col in enumerate(self.scheme):
                self._figs[f'{col}_predict'] = make_subplots(rows=1, cols=1, subplot_titles=("Prediction",))
                self._figs[f'{col}_predict'].update_xaxes(title_text='Index', row=1, col=1)
                self._figs[f'{col}_predict'].update_yaxes(title_text=col, row=1, col=1)
                self._figs[f'{col}_predict'].update_layout(showlegend=False)
        if update_fig:
            for i, col in enumerate(self.scheme):
                self._figs[f'{col}_predict'].add_trace(go.Scatter(x=X.index, y=X.loc[:, col], mode='lines', name=col, line=dict(color='blue')), row=1, col=1)
                self._figs[f'{col}_predict'].add_trace(go.Scatter(x=X.index[X["outliers"]], y=X.loc[X["outliers"], col], mode='markers', name='Outliers', marker=dict(color='red')), row=1, col=1)
            
if __name__ == "__main__":

    data = pd.read_excel("./data/pump_train.xlsx")
    ee = EllipticEnvelopeAnomalyDetection().fit(data)
    n_data = pd.read_excel("./data/pump_test.xlsx")
    y_pred = ee.predict(n_data)


    

    print("Done!")