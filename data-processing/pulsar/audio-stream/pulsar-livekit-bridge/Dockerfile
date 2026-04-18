FROM docker.io/rust:1.94.1-slim-trixie AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    g++ \
    libglib2.0-dev \
    libopus-dev \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
COPY schemas/ schemas/
COPY pulsar-livekit-bridge/Cargo.toml pulsar-livekit-bridge/build.rs pulsar-livekit-bridge/
COPY pulsar-livekit-bridge/src/ pulsar-livekit-bridge/src/
RUN cargo build --release --manifest-path pulsar-livekit-bridge/Cargo.toml

FROM docker.io/debian:trixie-slim AS release
WORKDIR /app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libopus0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/pulsar-livekit-bridge/target/release/pulsar-livekit-bridge ./
COPY pulsar-livekit-bridge/.env.development ./
COPY pulsar-livekit-bridge/.env.production ./
CMD ["./pulsar-livekit-bridge"]
