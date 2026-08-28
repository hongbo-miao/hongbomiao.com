FROM docker.io/python:3.14.7-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_PROJECT_ENVIRONMENT=/usr/local
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-editable

FROM docker.io/python:3.14.7-slim-trixie AS release
WORKDIR /usr/src/app
COPY --from=builder /usr/local/lib/python3.14/site-packages/ /usr/local/lib/python3.14/site-packages/
COPY src/ src/
CMD ["python", "src/main.py"]
