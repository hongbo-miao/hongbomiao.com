FROM debian:trixie-20260610

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates curl \
        build-essential pkg-config \
        libsoapysdr-dev \
        libclang-dev \
        cmake \
        protobuf-compiler \
    && rm --recursive --force /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --silent --show-error --fail https://sh.rustup.rs | sh -s -- -y --profile=minimal --default-toolchain=nightly

WORKDIR /src
