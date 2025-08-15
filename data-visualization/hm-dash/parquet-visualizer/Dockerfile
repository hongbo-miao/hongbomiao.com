FROM ghcr.io/astral-sh/uv:0.8.11-python3.12-bookworm-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  # Target the system Python environment
  UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

COPY ["pyproject.toml", "uv.lock", "./"]
RUN uv sync --frozen
COPY src/ ./src/
EXPOSE 8050

# Run the application with Gunicorn
# - workers: Number of worker processes (2-4 x NUM_CORES is recommended)
# - threads: Number of threads per worker
# - timeout: Worker timeout in seconds
# - bind: Bind to all interfaces on port 8050
CMD ["gunicorn", "--workers=4", "--threads=2", "--timeout=120", "--bind=0.0.0.0:8050", "src.main:server"]
