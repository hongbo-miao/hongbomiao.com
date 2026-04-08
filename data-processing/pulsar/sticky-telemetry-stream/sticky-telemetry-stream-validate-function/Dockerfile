FROM ghcr.io/astral-sh/uv:0.11.3-python3.14-trixie-slim AS builder
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PROJECT_ENVIRONMENT=/usr/local
COPY sticky-telemetry-stream-schema/ sticky-telemetry-stream-schema/
COPY ["sticky-telemetry-stream-validate-function/pyproject.toml", "sticky-telemetry-stream-validate-function/uv.lock", "sticky-telemetry-stream-validate-function/.python-version", "sticky-telemetry-stream-validate-function/"]
RUN uv sync --frozen --no-dev --no-editable --directory sticky-telemetry-stream-validate-function

FROM docker.io/python:3.14.3-slim-trixie AS release
WORKDIR /usr/src/app
COPY --from=builder /usr/local/lib/python3.14/site-packages/ /usr/local/lib/python3.14/site-packages/
COPY sticky-telemetry-stream-validate-function/src/ src/
CMD ["python", "src/main.py"]
