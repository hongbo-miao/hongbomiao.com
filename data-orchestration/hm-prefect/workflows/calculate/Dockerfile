FROM ghcr.io/astral-sh/uv:0.11.4-python3.14-trixie-slim AS builder
WORKDIR /opt/prefect/workflow
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  # Target the system Python environment
  UV_PROJECT_ENVIRONMENT=/usr/local
COPY ["pyproject.toml", "uv.lock", ".python-version", "./"]
RUN uv sync --frozen --no-dev

FROM harbor.hongbomiao.com/docker-hub-proxy-cache/python:3.14-slim-trixie AS release
WORKDIR /opt/prefect/workflow
COPY --from=builder /usr/local/lib/python3.14/site-packages/ /usr/local/lib/python3.14/site-packages/
COPY --from=builder /usr/local/bin/prefect /usr/local/bin/prefect
COPY ["src/", "/opt/prefect/workflow/src/"]
