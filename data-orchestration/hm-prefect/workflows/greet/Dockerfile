FROM docker.io/prefecthq/prefect:3.4.9-python3.12-kubernetes AS base
WORKDIR /opt/prefect/workflow

FROM base AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  # Target the system Python environment
  UV_PROJECT_ENVIRONMENT=/usr/local
COPY --from=ghcr.io/astral-sh/uv:0.8.2-python3.12-alpine /usr/local/bin/uv /bin/
COPY ["pyproject.toml", "uv.lock", "./"]
RUN uv sync --frozen

FROM base AS release
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY ["src/", "/opt/prefect/workflow/src/"]
