FROM ghcr.io/astral-sh/uv:0.11.2-python3.14-trixie-slim AS builder
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PROJECT_ENVIRONMENT=/usr/local
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ make && rm -rf /var/lib/apt/lists/*
COPY ["partitioned-telemetry-stream-subscriber/pyproject.toml", "partitioned-telemetry-stream-subscriber/uv.lock", "partitioned-telemetry-stream-subscriber/.python-version", "./"]
RUN uv sync --frozen --no-dev

FROM docker.io/python:3.14.3-slim-trixie AS release
WORKDIR /usr/src/app
COPY --from=builder /usr/local/lib/python3.14/site-packages/ /usr/local/lib/python3.14/site-packages/
COPY schemas/ ../schemas/
COPY partitioned-telemetry-stream-subscriber/src/ src/
CMD ["python", "src/main.py"]
