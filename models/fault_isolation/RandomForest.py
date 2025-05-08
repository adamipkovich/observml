import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.tree import plot_tree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from mlflow.client import MlflowClient
import distinctipy

class RandomForestModel:

    _model : RandomForestClassifier= None
    _std_scaler : StandardScaler = None
    _figs : dict[str, go.Figure] = dict()   
    _auto : bool = True
    _is_fitted : bool = False

    def __init__(self) -> None: ## auto
        self._std_scaler = StandardScaler()
        self._model = RandomForestClassifier()


    def fit(self, X, y=None):

        nX = X.drop(columns=["target"])
        self.scheme = nX.columns
        nX = self._std_scaler.fit_transform(nX)
        nX = pd.DataFrame(nX, columns=self.scheme)
        y = X["target"]
        
        hyper_params = {'n_estimators': [10, 50, 100],
                        'ccp_alpha': [0.1, 0.01, 0.001],
                        'max_depth': np.arange(2, 6, 1), }
        
        clf = GridSearchCV(RandomForestClassifier(), hyper_params, cv=5)
        clf.fit(nX, y) # train the model
        self._model = clf.best_estimator_
        self._score = clf.best_score_
        y_hat = clf.predict(nX)
        self._figs['performance'] = make_subplots(rows=1, cols=1, subplot_titles=("Confusion Matrix",))
        cm = confusion_matrix(y, y_hat, normalize='true')
        self._figs['performance'].add_trace(go.Heatmap(z=cm, x=self._model.classes_, y=self._model.classes_), row=1, col=1)
        self._figs['performance'].update_layout(title_text='Confusion Matrix', width=500, height=500, showlegend=False)
        self._figs['performance'].update_xaxes(title_text='Predicted', row=1, col=1)
        self._figs['performance'].update_yaxes(title_text='Actual', row=1, col=1)

        self._figs["importance"] = make_subplots(rows=1, cols=1, subplot_titles=("Classes",))
        self._figs['importance'].add_trace(go.Bar(x=nX.columns.to_list(), y=self._model.feature_importances_, name='Feature Importance'), row=1, col=1)
        self._figs['importance'].update_layout(title_text='Feature Importance', width=500, height=500, showlegend=False)
        self._figs['importance'].update_xaxes(title_text='Feature', row=1, col=1)
        self._figs['importance'].update_yaxes(title_text='Importance', row=1, col=1)

        colors = ["rgb" + str(c) for c in distinctipy.get_colors(X["target"].nunique())]
        
        self.colors = {}
        for i, cl in enumerate(X["target"].unique()):
            self.colors[cl] = colors[i]

        for j, col in enumerate(self.scheme):
                rX = X.loc[:, col]

                self._figs[f'{col}_classes'] = make_subplots(rows=1, cols=1, subplot_titles=("Classes",))
                self._figs[f'{col}_classes'].update_xaxes(title_text='Time', row=1, col=1)
                self._figs[f'{col}_classes'].update_yaxes(title_text= col, row=1, col=1)
                self._figs[f'{col}_classes'].update_layout(showlegend=False)

                for i, cl in enumerate(self.colors.keys()):
                    fig_d =  rX.loc[y == cl]
                    ds = fig_d.index.to_list()
                    sub_col = self.colors[cl]
                    self._figs[f'{col}_classes'].add_trace(go.Scatter(x=ds, y=fig_d, mode='markers', name=str(cl), marker=dict(color=sub_col)), row=1, col=1)

        
        self.update_predict(X, reset_fig=True, update_fig=False)

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise Exception("Model not fitted. Please fit the model first.")
        if "target" in X.columns:
            nX = X.drop(columns=["target"])
        else:
            nX = X.copy()
        nX = self._std_scaler.transform(nX.loc[:, self.scheme])
        nX = pd.DataFrame(nX, columns=self.scheme)
        y_hat = self._model.predict(nX)
        X["y_pred"] = y_hat
        return X
    

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
    #mlflow.set_tracking_uri("http://localhost:5000")
    data = pd.read_excel("./data/detect_train.xlsx")
    dt = RandomForestModel()

    X = data.rename(columns={"Output (S)": "target", "ds": "ds"})
    dt.fit(X)
    
    dt._figs["performance"].show()
    dt._figs["importance"].show()
    dt._figs["classes"].show()

    nX = pd.read_excel("./data/detect_test.xlsx")
    nX = data.rename(columns={"Output (S)": "target", "ds": "ds"})
    y = dt.predict(nX)
    dt.update_predict(y)
    dt._figs["prediction"].show()
    #n_data = pd.read_excel("./data/detect_test.xlsx")
    print("Done!")

