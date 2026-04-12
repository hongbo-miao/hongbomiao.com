FROM docker.io/python:3.14.4-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy
RUN apt-get update && \
  apt-get install --yes --no-install-recommends default-jdk-headless build-essential && \
  rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java
COPY ["pyproject.toml", "uv.lock", ".python-version", "./"]
RUN uv sync --frozen --no-dev --no-editable

FROM docker.io/python:3.14.4-slim-trixie AS release
RUN apt-get update && \
  apt-get install --yes --no-install-recommends default-jdk-headless curl && \
  rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app
COPY --from=builder /usr/src/app/.venv/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
RUN mkdir -p /app/lib && \
  curl --silent --fail --show-error --location \
  --output /app/lib/flink-sql-connector-pulsar-4.1.0-1.18.jar \
  https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-pulsar/4.1.0-1.18/flink-sql-connector-pulsar-4.1.0-1.18.jar && \
  curl --silent --fail --show-error --location \
  --output /app/lib/flink-sql-avro-1.20.3.jar \
  https://repo1.maven.org/maven2/org/apache/flink/flink-sql-avro/1.20.3/flink-sql-avro-1.20.3.jar
COPY src/ src/
ENV JAVA_HOME=/usr/lib/jvm/default-java
CMD ["python", "src/main.py"]
