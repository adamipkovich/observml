FROM python:3.11.9-slim

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

ENTRYPOINT ["mlflow", "server","--host", "0.0.0.0", "--port", "5000"]