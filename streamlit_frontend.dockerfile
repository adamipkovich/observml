FROM python:3.12.4-slim

WORKDIR /app
RUN apt-get update && apt-get install -y 
#    build-essential \
#    curl \
#    software-properties-common \
#    git \
#    && rm -rf /var/lib/apt/lists/*
# RUN git clone https://github.com/streamlit/streamlit-example.git . -> we can pull the repo if any changes happened...

COPY . /app/

RUN pip3 install --no-cache-dir --upgrade pip 
RUN pip3 install --no-cache-dir requests
RUN pip3 install --no-cache-dir plotly
RUN pip3 install --no-cache-dir streamlit

# This command creates a .streamlit directory in the home directory of the container.
RUN mkdir ~/.streamlit

# This copies your Streamlit configuration file into the .streamlit directory you just created.
RUN cp config.toml ~/.streamlit/config.toml

# Similar to the previous step, this copies your Streamlit credentials file into the .streamlit directory.
RUN cp credentials.toml ~/.streamlit/credentials.toml


ENV HUB_URL "http://0.0.0.0:8010"
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD [ "streamlit_frontend.py" ]

#CMD ["streamlit_frontend.py", "--server.port=8501", "--server.address=0.0.0.0" ]

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
#CMD  ["streamlit_frontend.py", "--server.port=8501", f"--server.address=0.0.0.0"] 