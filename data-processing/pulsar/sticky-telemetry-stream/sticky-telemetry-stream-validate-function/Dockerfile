FROM docker.io/python:3.14.4-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.5 /uv /usr/local/bin/uv
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PROJECT_ENVIRONMENT=/usr/local
COPY sticky-telemetry-stream-schema/ sticky-telemetry-stream-schema/
COPY ["sticky-telemetry-stream-validate-function/pyproject.toml", "sticky-telemetry-stream-validate-function/uv.lock", "sticky-telemetry-stream-validate-function/.python-version", "sticky-telemetry-stream-validate-function/"]
RUN uv sync --frozen --no-dev --no-editable --directory sticky-telemetry-stream-validate-function

FROM docker.io/python:3.14.4-slim-trixie AS release
WORKDIR /usr/src/app
RUN apt-get update && apt-get install --yes --no-install-recommends protobuf-compiler && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.14/site-packages/ /usr/local/lib/python3.14/site-packages/
COPY proto/ proto/
COPY sticky-telemetry-stream-validate-function/src/ src/
RUN mkdir -p src/shared/telemetry/proto && protoc --proto_path=proto --python_out=src/shared/telemetry/proto proto/telemetry_record.proto
CMD ["python", "src/main.py"]
