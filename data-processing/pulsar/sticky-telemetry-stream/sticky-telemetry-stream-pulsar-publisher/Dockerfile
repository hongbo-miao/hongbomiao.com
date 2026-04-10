FROM docker.io/python:3.14.4-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.5 /uv /uvx /bin/
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PROJECT_ENVIRONMENT=/usr/local
COPY sticky-telemetry-stream-schema/ sticky-telemetry-stream-schema/
COPY ["sticky-telemetry-stream-pulsar-publisher/pyproject.toml", "sticky-telemetry-stream-pulsar-publisher/uv.lock", "sticky-telemetry-stream-pulsar-publisher/.python-version", "sticky-telemetry-stream-pulsar-publisher/"]
RUN uv sync --frozen --no-dev --no-editable --directory sticky-telemetry-stream-pulsar-publisher

FROM docker.io/python:3.14.4-slim-trixie AS release
WORKDIR /usr/src/app
COPY --from=builder /usr/local/lib/python3.14/site-packages/ /usr/local/lib/python3.14/site-packages/
COPY sticky-telemetry-stream-pulsar-publisher/src/ src/
CMD ["python", "src/main.py"]
