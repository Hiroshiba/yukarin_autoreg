FROM hiroshiba/hiho-deep-docker-base:chainer7.1-cuda9.0

RUN apt-get update && \
    apt-get install -y swig libsndfile1-dev libasound2-dev && \
    apt-get clean
RUN conda install -y cython numpy numba

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY requirements-dev.txt /app/
RUN pip install -r requirements-dev.txt

# cpp
COPY src_cython /app/src_cython
RUN cd /app/src_cython && \
    curl https://raw.githubusercontent.com/Hiroshiba/yukarin_autoreg_cpp/cuda9.0/CppWaveRNN/CppWaveRNN.h > /app/src_cython/CppWaveRNN.h && \
    CFLAGS="-I." \
    LDFLAGS="-L." \
    python setup.py install
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/src_cython"

# optuna
RUN apt-get update && \
    apt-get install -y python3-dev libmysqlclient-dev && \
    apt-get clean && \
    pip install optuna mysqlclient

CMD bash
