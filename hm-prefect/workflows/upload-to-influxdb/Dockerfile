FROM docker.io/python:3.11.3 AS base
WORKDIR /usr/src/app
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1


FROM base AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir poetry \
  && python -m venv /venv/
COPY ["hm-prefect/workflows/upload-to-influxdb/pyproject.toml", "hm-prefect/workflows/upload-to-influxdb/poetry.lock", "./"]
RUN . /venv/bin/activate \
  && poetry install --only=main --no-root
COPY ["hm-prefect/workflows/upload-to-influxdb/", "./"]
RUN . /venv/bin/activate \
  && poetry build


FROM docker.io/prefecthq/prefect:2.10.4-python3.11-kubernetes AS release
# WORKDIR /opt/prefect
COPY --from=builder /venv/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY ["hm-prefect/workflows/upload-to-influxdb/src/", "./flows/src/"]
