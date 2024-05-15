FROM docker.io/ubuntu:noble
WORKDIR /usr/src/app
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# https://pgbackrest.org/user-guide.html
# hadolint ignore=DL3015
RUN mkdir --parents /usr/src/app/build \
  && apt-get update --yes \
  && apt-get install --yes wget make gcc libpq-dev libssl-dev libxml2-dev pkg-config liblz4-dev libzstd-dev libbz2-dev libz-dev libyaml-dev \
  && apt-get clean \
  && rm -r -f /var/lib/apt/lists/* \
  && wget --quiet --output-document=- https://github.com/pgbackrest/pgbackrest/archive/release/2.43.tar.gz | tar -x -z -C /usr/src/app/build \
  && cd /usr/src/app/build/pgbackrest-release-2.43/src/ \
  && ./configure \
  && make
