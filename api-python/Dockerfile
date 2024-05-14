FROM docker.io/python:3.12.3-alpine AS base
WORKDIR /usr/src/app
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1


FROM base AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1
ARG LIBRDKAFKA_VERSION=2.4.0
# bash, g++, make are for librdkafka, which is for confluent-kafka
# gcc, libffi-dev, musl-dev are for Poetry
# hadolint ignore=DL3003
RUN apk add --no-cache gcc libffi-dev musl-dev bash g++ make \
  # Install Poetry
  && pip install --no-cache-dir poetry \
  && python -m venv /venv/ \
  # Install librdkafka
  # https://github.com/edenhill/librdkafka#build-from-source
  && wget --no-verbose --output-document=librdkafka.tar.gz https://github.com/confluentinc/librdkafka/archive/refs/tags/v${LIBRDKAFKA_VERSION}.tar.gz \
  && tar --extract --file=librdkafka.tar.gz \
  && rm -f librdkafka.tar.gz \
  && cd librdkafka-${LIBRDKAFKA_VERSION} \
  && ./configure \
  && make \
  && make install \
  && cd .. \
  && rm -r -f librdkafka-${LIBRDKAFKA_VERSION}/
COPY ["api-python/pyproject.toml", "api-python/poetry.lock", "./"]
RUN . /venv/bin/activate \
  && poetry install --only=main --no-root
COPY api-python ./
RUN . /venv/bin/activate \
  && poetry build


FROM base AS release
COPY --from=builder /venv/ /venv/
COPY --from=builder /usr/src/app/dist/ ./
# librdkafka related files in /usr/local/lib/
# - librdkafka++.a
# - librdkafka++.so
# - librdkafka++.so.1
# - librdkafka-static.a
# - librdkafka.a
# - librdkafka.so
# - librdkafka.so.1
COPY --from=builder /usr/local/lib/ /usr/local/lib/
RUN . /venv/bin/activate \
  && pip install --no-cache-dir ./*.whl
COPY ["api-python/docker-entrypoint.sh", "./"]
EXPOSE 35903
ENTRYPOINT ["./docker-entrypoint.sh"]
