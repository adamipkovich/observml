from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

import os

from PIL import Image

from sklearn.base import BaseEstimator
import pandas as pd
import plotly.graph_objects as go



class HeuristicsMiner(BaseEstimator):
    """Heuristics Miner model using pm4py library. Can only be used if package is provided as an open source licence."""
    _model = None
    _figs : dict[str, go.Figure] = {}  

    def __init__(self, * , mac = 10) -> None:
        self.mac = mac
        pass

    def fit(self, X: pd.DataFrame, y = None):
        self.predict(X)
        return self
        

    def predict(self, X: pd.DataFrame):
        self._model = heuristics_miner.apply_heu(X, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: self.mac})
        gviz = hn_visualizer.apply(self._model, parameters={'format': 'png'})
        hn_visualizer.save(gviz, './proc.png')
        img = Image.open('./proc.png')
        self._figs["HeuristicsNet"] = go.Figure()
        self._figs["HeuristicsNet"].add_trace(
            go.Scatter(
                x=[0, img.size[0]],
                y=[0, img.size[1]],
                mode="markers",
                marker_opacity=0
            )
        )

        # Configure axes
        self._figs["HeuristicsNet"].update_xaxes(
            visible=False,
            range=[0, img.size[0]]
        )

        self._figs["HeuristicsNet"].update_yaxes(
            visible=False,
            range=[0, img.size[1]],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x"
        )
        self._figs["HeuristicsNet"].add_layout_image(
            dict(
                x=0,
                sizex=img.size[0] ,
                y=img.size[1],
                sizey=img.size[1],
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=img)
        )
        self._figs["HeuristicsNet"].update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        self._figs["HeuristicsNet"].update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        self._figs["HeuristicsNet"].update_layout(title='Heuristics Miner')
        self._figs["HeuristicsNet"].update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', autosize = True, paper_bgcolor='rgba(0,0,0,0)', width = img.size[0], height = img.size[1])
        
        return 
    
if __name__ == "__main__":
    data = pd.read_csv("data/cable_head_mach_27.csv")
    hm = HeuristicsMiner()
    hm.fit(data)
    hm._figs["HeuristicsNet"].show()

    print("Done!")
