# pull official base image
FROM python:3.8-slim-buster

# set wrok directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . /usr/src/app/
# install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt