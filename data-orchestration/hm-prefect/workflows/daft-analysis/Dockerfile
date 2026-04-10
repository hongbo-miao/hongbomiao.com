FROM docker.io/python:3.13.13-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/
WORKDIR /opt/prefect/workflow
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  # Target the system Python environment
  UV_PROJECT_ENVIRONMENT=/usr/local
COPY ["pyproject.toml", "uv.lock", ".python-version", "./"]
RUN uv sync --frozen --no-dev

FROM docker.io/python:3.13.13-slim-trixie AS release
WORKDIR /opt/prefect/workflow
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/prefect /usr/local/bin/prefect
COPY ["src/", "/opt/prefect/workflow/src/"]
