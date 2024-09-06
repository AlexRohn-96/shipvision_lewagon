FROM python:3.10.6-buster
COPY shipvision_backend / shipvision_backend
COPY api / api
COPY models / models
COPY requirements.txt /requirements.txt

# Copy the credentials file into the /app directory
COPY wagon-429513-bffb398f06b4.json /wagon-429513-bffb398f06b4.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
