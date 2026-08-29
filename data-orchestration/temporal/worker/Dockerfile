ARG RUST_VERSION=1.93

FROM docker.io/rust:${RUST_VERSION}-trixie AS builder

# The Temporal SDK's proto types are generated from .proto files in a build
# script, so protoc has to be on the builder even though nothing needs it at
# runtime.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends protobuf-compiler libprotobuf-dev \
    && rm --recursive --force /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN cargo build --locked --release --package=worker

FROM docker.io/debian:trixie-slim

# The Temporal client dials the frontend over gRPC, which may be TLS-wrapped
# in production; the system CA bundle is needed to validate certificates.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm --recursive --force /var/lib/apt/lists/*

COPY --from=builder /build/target/release/worker /usr/local/bin/worker

ENTRYPOINT ["/usr/local/bin/worker"]
