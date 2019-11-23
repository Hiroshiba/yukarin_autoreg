FROM hiroshiba/hiho-deep-docker-base:chainer6-cuda9.0

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# optuna
RUN apt-get update
RUN apt-get install -y python3-dev libmysqlclient-dev
RUN pip install optuna mysqlclient

# add applications
COPY utility /app/utility
COPY tests /app/tests
COPY yukarin_autoreg /app/yukarin_autoreg
COPY scripts /app/scripts
COPY train*.py /app/

CMD bash
