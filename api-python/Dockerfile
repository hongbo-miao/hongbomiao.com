FROM python:3.11-alpine AS base
WORKDIR /usr/src/app
ENV PYTHONFAULTHANDLER=1 \
  PYTHONHASHSEED=random \
  PYTHONUNBUFFERED=1


FROM base AS builder
ENV PIP_DEFAULT_TIMEOUT=100 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1
RUN apk add --no-cache gcc libffi-dev musl-dev \
  && pip install --no-cache-dir poetry \
  && python -m venv /venv/
COPY ["api-python/pyproject.toml", "api-python/poetry.lock", "./"]
RUN . /venv/bin/activate && \
  poetry install --no-dev --no-root
COPY api-python ./
RUN . /venv/bin/activate && \
  poetry build


FROM base AS release
COPY --from=builder /venv/ /venv/
COPY --from=builder /usr/src/app/dist/ ./
RUN . /venv/bin/activate && \
  pip install --no-cache-dir ./*.whl
COPY ["api-python/docker-entrypoint.sh", "./"]
ENTRYPOINT ["./docker-entrypoint.sh"]
EXPOSE 35903
