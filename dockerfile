FROM python:3.8.0

RUN apt-get update

RUN pip install --upgrade pip &&\
  pip install tensorflow &&\
  pip install h5py &&\
  pip install opencv-python