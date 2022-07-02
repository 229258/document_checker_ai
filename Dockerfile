#FROM node:alpine
#FROM python:3.9.0-alpine
FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED=True
#RUN apk add --no-cache python3 py3-pip
RUN ln -sf python3 /usr/bin/python
# RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
#RUN python3 -m ensurepip
#RUN pip3 install --no-cache --upgrade pip setuptools
# RUN python -m venv app

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN python -m pip install --upgrade Pillow
RUN python -m pip install tensorflow

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY . /app

CMD ["./start.sh"]
# CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
