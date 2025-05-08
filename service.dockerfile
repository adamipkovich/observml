FROM python:3.11.9-slim

COPY . .


## For SPMF
RUN apt-get update

RUN apt-get install -y wget apt-transport-https gnupg
RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
RUN echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update
RUN apt-get install -y temurin-17-jdk

## this may olve the problem with unattainable graphviz executables
RUN apt-get install -y graphviz  

RUN pip3 install --no-cache-dir pydot graphviz
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir tensorflow 
## for pyagrum
#RUN apt-get install libgomp1 -y

ENV MLFLOW_URI="http://0.0.0.0:5000"
ENV RABBIT_HOST="0.0.0.0"
ENV RABBIT_PORT "5672"

ENV RABBIT_USER "guest"
ENV RABBIT_PASSWORD "guest"

EXPOSE 8010

CMD ["uvicorn", "ExperimentHubAPI:app", "--host", "0.0.0.0", "--port", "8010"]

#ENTRYPOINT ["fastapi", "run", "ExperimentHubAPI.py"]