FROM docker.io/python:3.11.15-slim-trixie AS builder
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

FROM flink:1.20-java11 AS release
RUN apt-get update && \
  apt-get install --yes --no-install-recommends python3.11 python3-setuptools curl && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
  rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/.venv/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages
RUN curl --silent --fail --show-error --location \
  --output /opt/flink/lib/flink-sql-connector-pulsar-4.1.0-1.18.jar \
  https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-pulsar/4.1.0-1.18/flink-sql-connector-pulsar-4.1.0-1.18.jar && \
  curl --silent --fail --show-error --location \
  --output /opt/flink/lib/flink-sql-avro-1.20.3.jar \
  https://repo1.maven.org/maven2/org/apache/flink/flink-sql-avro/1.20.3/flink-sql-avro-1.20.3.jar
COPY src/ /app/src/
