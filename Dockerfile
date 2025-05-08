FROM python:3.12-slim

WORKDIR /housing-model

COPY ./requirements.txt .
COPY src/server.py .

RUN apt-get update \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && apt-get remove -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

CMD ["gunicorn", "server:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80"]
