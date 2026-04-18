FROM docker.io/rust:1.94.1-slim-trixie AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
COPY schemas/ schemas/
COPY audio-stream-publisher/Cargo.toml audio-stream-publisher/build.rs audio-stream-publisher/
COPY audio-stream-publisher/src/ audio-stream-publisher/src/
RUN cargo build --release --manifest-path audio-stream-publisher/Cargo.toml

FROM docker.io/debian:trixie-slim AS release
WORKDIR /app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/audio-stream-publisher/target/release/audio-stream-publisher ./
COPY audio-stream-publisher/data/ ./data/
COPY audio-stream-publisher/.env.development ./
COPY audio-stream-publisher/.env.production ./
CMD ["./audio-stream-publisher"]
