FROM python:3.10

RUN pip install --upgrade pip
RUN pip install mlflow
WORKDIR /workspace

EXPOSE 5000
