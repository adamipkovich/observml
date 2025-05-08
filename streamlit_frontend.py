import sys
import os
import plotly.graph_objects as go
import streamlit as st
import requests
import time
import json
from plotly.io import from_json

try:
    host = os.environ['HUB_URL']  #  sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000" # -- get from command line #
except KeyError:
    host = "http://localhost:8010"

if st.session_state.get('refresh_dur') is None:
    st.session_state['refresh_dur'] = 5

if 'experiments' not in st.session_state:  ## relevant experiments
    st.session_state['experiments'] = {"None" : [["None"], ["None"]]}

response = requests.get(f"{host}/experiments")
if response.status_code == 200:
    data = json.loads(response.content)
    st.session_state['experiments'] = data


st.set_page_config(
    page_title="Experiment Hub User Interface",
    layout="wide"
)


@st.cache_resource
def get_cfg(experiment):
    if experiment == 'None':
        return None
    data = requests.get(f"{host}/{experiment}/cfg")
    #d = pd.read_json(data.content.decode('utf-8'))     
    return data.content  #StreamlitRenderer(d, spec="./gw_config.json", spec_io_mode="rw")

with st.sidebar:
    mode = st.radio("Select Mode", ["EDA", "Monitor"], captions=["Exploratory Data Analysis", "Monitor"])
    st.write("Refresh Duration: ")
    st.slider("Refresh Duration", 1, 60, 1, key='refresh_dur')
    current_exp = st.selectbox("Experiment", list(st.session_state['experiments'].keys()))

if mode == "EDA":

    exp_col, rep_col = st.columns(2)   

    with exp_col:
        st.container()
    with rep_col:
        
        current_eda = st.selectbox("EDA Report", st.session_state['experiments'][current_exp][1])

    fig_cont = st.container()
    with fig_cont:
            if current_exp != 'None' and current_eda  != 'None':
                response = requests.get(f"{host}/{current_exp}/plot_eda/{current_eda}")   
                fig = from_json(response.content)
                if isinstance(fig.data[0], go.Heatmap):                
                    fig.update_yaxes(autorange = "reversed")
                fig.update_layout(
                    font = dict(
                        size = 18,
                        color = "black" 
                    ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(go.Figure(), use_container_width=True)

elif mode == "Monitor":
    exp_col, rep_col = st.columns(2)    

    with exp_col:
         st.write("Model Metadata: ")
         st.dataframe(data =json.loads(get_cfg(current_exp)) )
    
    with rep_col:
        st.write("Report: ")
        current_plot = st.selectbox("Report", st.session_state['experiments'][current_exp][0])

    fig_cont = st.container()
    with fig_cont:
            if current_exp != 'None' and current_plot != 'None':
                response = requests.get(f"{host}/{current_exp}/plot/{current_plot}")   
                fig = from_json(response.content)
                if isinstance(fig.data[0], go.Heatmap):                
                    fig.update_yaxes(autorange = "reversed")
                
                fig.update_layout(
                    font = dict(
                        size = 18,
                        color = "black" 
                    ))    
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(go.Figure(), use_container_width=True)


else:
    st.write("Invalid Mode Selected.")
    st.stop()


time.sleep(st.session_state['refresh_dur'])
st.rerun() 
##TODO: maybe add current_fig tp session state to avoid pulling extensive pressure on backend.
