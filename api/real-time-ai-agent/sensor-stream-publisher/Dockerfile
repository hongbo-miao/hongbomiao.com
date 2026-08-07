ARG RUST_VERSION=1.93

FROM docker.io/rust:${RUST_VERSION}-trixie AS builder

# fluss-rs generates its RPC types from .proto files in its build script, so
# protoc has to be on the builder even though nothing needs it at runtime.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends protobuf-compiler \
    && rm --recursive --force /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN cargo build --locked --release

FROM docker.io/debian:trixie-slim

# fluss-rs dials the coordinator over TLS-capable rustls, which still needs
# the system CA bundle to validate certificates when TLS is turned on.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm --recursive --force /var/lib/apt/lists/*

COPY --from=builder /build/target/release/sensor-stream-publisher /usr/local/bin/sensor-stream-publisher

ENTRYPOINT ["/usr/local/bin/sensor-stream-publisher"]
