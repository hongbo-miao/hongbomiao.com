FROM docker.io/rust:1.94.1-slim-trixie AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    g++ \
    libopus-dev \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
COPY schemas/ schemas/
COPY grpc-pulsar-bridge/Cargo.toml grpc-pulsar-bridge/build.rs grpc-pulsar-bridge/
COPY grpc-pulsar-bridge/src/ grpc-pulsar-bridge/src/
RUN cargo build --release --manifest-path grpc-pulsar-bridge/Cargo.toml

FROM docker.io/debian:trixie-slim AS release
WORKDIR /app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    libopus0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/grpc-pulsar-bridge/target/release/grpc-pulsar-bridge ./
COPY grpc-pulsar-bridge/.env.development ./
COPY grpc-pulsar-bridge/.env.production ./
CMD ["./grpc-pulsar-bridge"]
