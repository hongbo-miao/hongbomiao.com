FROM docker.io/python:3.14.4-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.11.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml pytest.ini ./
RUN uv sync
COPY . .
