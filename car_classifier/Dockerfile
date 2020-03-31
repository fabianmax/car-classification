#FROM python:3.8
FROM tensorflow/tensorflow:latest-py3

EXPOSE 5000

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
ADD . /app

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "grad_cam_api:app"]
