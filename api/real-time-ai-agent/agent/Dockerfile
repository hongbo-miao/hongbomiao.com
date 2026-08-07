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

# The Anthropic client dials api.anthropic.com over TLS, which needs the
# system CA bundle to validate the certificate.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm --recursive --force /var/lib/apt/lists/*

COPY --from=builder /build/target/release/agent /usr/local/bin/agent

EXPOSE 8083
ENTRYPOINT ["/usr/local/bin/agent"]
