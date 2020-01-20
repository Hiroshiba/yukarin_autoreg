FROM hiroshiba/hiho-deep-docker-base:chainer6-cuda9.0

RUN apt-get update && apt-get install -y swig libsndfile1-dev libasound2-dev && apt-get clean
RUN conda install -y cython numpy numba

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY requirements-dev.txt /app/
RUN pip install -r requirements-dev.txt

# optuna
RUN apt-get update
RUN apt-get install -y python3-dev libmysqlclient-dev
RUN pip install optuna mysqlclient

CMD bash
