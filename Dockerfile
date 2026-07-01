FROM jupyter/pyspark-notebook:latest
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
