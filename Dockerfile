# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/base-notebook:python-3.8.4
FROM $BASE_CONTAINER

LABEL maintainer="FumiyukiKato <fumilemon79@gmail.com>"

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    emacs-nox \
    vim-tiny \
    git \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    python3-dev \
    # ---- nbconvert dependencies ----
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    # ----
    tzdata \
    unzip \
    nano \
    curl \
    gfortran \
    liblapack-dev \
    libblas-dev \
    libpq-dev \
    libncurses5-dev \
    swig \
    gnumeric \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# python
COPY requirements.txt /home/jovyan/work/requirements.txt
RUN pip install -r /home/jovyan/work/requirements.txt

RUN jupyter nbextension enable --py widgetsnbextension

ENV PYTHONPATH=/home/jovyan/work/src:/home/jovyan/work/src/ektelo:/home/jovyan/work/competitors/hdmm/src

WORKDIR /home/jovyan/work
