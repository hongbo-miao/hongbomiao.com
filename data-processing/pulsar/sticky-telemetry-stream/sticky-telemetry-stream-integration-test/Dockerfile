FROM docker.io/python:3.14.4-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml pytest.ini ./
RUN uv sync
COPY . .
