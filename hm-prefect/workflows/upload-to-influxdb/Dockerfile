FROM docker.io/python:3.11.3-alpine AS base
WORKDIR /usr/src/app
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1


FROM base AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1
# gcc, libffi-dev, musl-dev are for Poetry
# hadolint ignore=DL3003
RUN apk add --no-cache gcc libffi-dev musl-dev \
  # Install Poetry
  && pip install --no-cache-dir poetry \
  && python -m venv /venv/
COPY ["hm-prefect/workflows/upload-to-influxdb/pyproject.toml", "hm-prefect/workflows/upload-to-influxdb/poetry.lock", "./"]
RUN . /venv/bin/activate \
  && poetry install --only=main --no-root
RUN . /venv/bin/activate \
  && poetry build


FROM docker.io/prefecthq/prefect:2.10.1-python3.11-kubernetes AS release
COPY --from=builder /venv/ /venv/
COPY --from=builder /usr/src/app/dist/ ./
RUN . /venv/bin/activate \
  && pip install --no-cache-dir ./*.whl
COPY ["hm-prefect/workflows/upload-to-influxdb/src/", "/opt/prefect/flows/src/"]
