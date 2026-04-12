FROM docker.io/python:3.14.4-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PROJECT_ENVIRONMENT=/usr/local
COPY ["pyproject.toml", "uv.lock", ".python-version", "./"]
RUN uv sync --frozen --no-dev

FROM docker.io/python:3.14.4-slim-trixie AS release
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY src/ ./src/
EXPOSE 8050
CMD ["gunicorn", "--workers=4", "--threads=2", "--timeout=120", "--bind=0.0.0.0:8050", "src.main:server"]
