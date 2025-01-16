FROM docker.io/python:3.13.1-alpine AS base
WORKDIR /usr/src/app
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONOPTIMIZE=2 \
  PYTHONDONTWRITEBYTECODE=1

FROM base AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy
ARG LIBRDKAFKA_VERSION=2.6.1
# gcc, g++, make, bash are for building librdkafka
# hadolint ignore=DL3003
RUN apk add --no-cache gcc g++ make bash \
  && python -m venv /venv/ \
  # Install librdkafka
  # https://github.com/confluentinc/librdkafka#build-from-source
  && wget --no-verbose --output-document=librdkafka.tar.gz https://github.com/confluentinc/librdkafka/archive/refs/tags/v${LIBRDKAFKA_VERSION}.tar.gz \
  && tar --extract --file=librdkafka.tar.gz \
  && rm -f librdkafka.tar.gz \
  && cd librdkafka-${LIBRDKAFKA_VERSION} \
  && ./configure \
  && make \
  && make install \
  && cd .. \
  && rm -r -f librdkafka-${LIBRDKAFKA_VERSION}/
COPY --from=ghcr.io/astral-sh/uv:0.5.20-python3.13-alpine /usr/local/bin/uv /bin/
COPY ["api-python/pyproject.toml", "api-python/uv.lock", "./"]
RUN . /venv/bin/activate \
  && uv sync --frozen
COPY api-python ./
RUN . /venv/bin/activate \
  && uv build \
  && pip install --no-cache-dir dist/*.whl \
  && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels dist/*.whl \
  && apk del gcc g++ make bash

FROM base AS release
COPY --from=builder /venv/ /venv/
# librdkafka related files in /usr/local/lib/
# - librdkafka++.a
# - librdkafka++.so
# - librdkafka++.so.1
# - librdkafka-static.a
# - librdkafka.a
# - librdkafka.so
# - librdkafka.so.1
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /wheels/ /wheels/
RUN . /venv/bin/activate \
  && pip install --no-cache-dir /wheels/*.whl \
  && rm -r -f /wheels
COPY ["api-python/", "./"]
EXPOSE 35903
ENTRYPOINT ["./docker-entrypoint.sh"]
