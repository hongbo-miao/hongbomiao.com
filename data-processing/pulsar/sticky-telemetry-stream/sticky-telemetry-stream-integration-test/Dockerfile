FROM ghcr.io/astral-sh/uv:0.11.4-python3.14-trixie-slim
WORKDIR /app
COPY pyproject.toml pytest.ini ./
RUN uv sync
COPY . .
