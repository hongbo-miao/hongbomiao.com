FROM python:3.11.0-alpine AS base
WORKDIR /usr/src/app
ENV PYTHONFAULTHANDLER=1 \
  PYTHONHASHSEED=random \
  PYTHONUNBUFFERED=1


FROM base AS builder
ENV PIP_DEFAULT_TIMEOUT=100 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1
# bash, g++, make are for librdkafka, which is for confluent-kafka
# gcc, libffi-dev, musl-dev are for Poetry
RUN apk add --no-cache gcc libffi-dev musl-dev bash g++ make \
  # Install Poetry
  && pip install --no-cache-dir poetry \
  && python -m venv /venv/ \
  # Install librdkafka
  # https://github.com/edenhill/librdkafka#build-from-source
  && wget https://github.com/edenhill/librdkafka/archive/refs/tags/v1.9.2.tar.gz \
  && tar xvzf v1.9.2.tar.gz \
  && cd librdkafka-1.9.2 \
  && ./configure \
  && make \
  && make install \
  && rm -rf librdkafka-1.9.2
COPY ["api-python/pyproject.toml", "api-python/poetry.lock", "./"]
RUN . /venv/bin/activate \
  && poetry install --only=main --no-root
COPY api-python ./
RUN . /venv/bin/activate \
  && poetry build


FROM base AS release
COPY --from=builder /venv/ /venv/
COPY --from=builder /usr/src/app/dist/ ./
RUN . /venv/bin/activate \
  && pip install --no-cache-dir ./*.whl
COPY ["api-python/docker-entrypoint.sh", "./"]
ENTRYPOINT ["./docker-entrypoint.sh"]
EXPOSE 35903
