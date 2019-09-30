FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y curl git bzip2 ffmpeg gcc g++ cmake

ENV LC_ALL C.UTF-8

# pyenv, python
ENV PYENV_ROOT /.pyenv
RUN curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)" && \
    pyenv install miniconda3-latest && \
    pyenv global miniconda3-latest && \
    pyenv rehash

RUN conda install -y python=3.7.3 numpy cython

# for ChainerX
ENV CHAINER_BUILD_CHAINERX 1
ENV CHAINERX_BUILD_CUDA 1
ENV CUDNN_ROOT_DIR /usr/local/cudnn-9.0-v7.0
ENV MAKEFLAGS -j8

# install requirements
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# add applications
COPY utility /app/utility
COPY tests /app/tests
COPY yukarin_autoreg /app/yukarin_autoreg
COPY scripts /app/scripts
COPY train*.py /app/

# optuna
RUN apt-get install -y python3-dev libmysqlclient-dev
RUN pip install optuna mysqlclient

CMD bash
