FROM docker.io/python:3.14.4-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PROJECT_ENVIRONMENT=/usr/local
RUN apt-get update && apt-get install --yes --no-install-recommends cmake g++ make && rm -rf /var/lib/apt/lists/*
COPY ["sticky-telemetry-stream-publisher/pyproject.toml", "sticky-telemetry-stream-publisher/uv.lock", "sticky-telemetry-stream-publisher/.python-version", "./"]
RUN uv sync --frozen --no-dev

FROM docker.io/python:3.14.4-slim-trixie AS release
WORKDIR /usr/src/app
COPY --from=builder /usr/local/lib/python3.14/site-packages/ /usr/local/lib/python3.14/site-packages/
COPY schemas/ ../schemas/
COPY sticky-telemetry-stream-publisher/src/ src/
CMD ["python", "src/main.py"]
