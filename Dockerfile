FROM hiroshiba/hiho-deep-docker-base:ubuntu18.04-chainer7.7-cuda10.2
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y swig libsndfile1-dev libasound2-dev && \
    apt-get clean
RUN conda install -y cython numpy numba

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r requirements.txt | grep -v 'chainer' | grep -v 'cupy'
COPY requirements-dev.txt /app/
RUN pip install -r requirements-dev.txt

# cpp
COPY src_cython /app/src_cython
RUN cd /app/src_cython && \
    CFLAGS="-I." LDFLAGS="-L." python setup.py install
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/src_cython"

# optuna
RUN apt-get update && \
    apt-get install -y python3-dev libmysqlclient-dev && \
    apt-get clean && \
    pip install optuna mysqlclient

CMD bash
