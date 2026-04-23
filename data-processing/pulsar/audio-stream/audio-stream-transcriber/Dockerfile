FROM docker.io/rust:1.94.1-slim-trixie AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    cmake g++ libglib2.0-dev libopus-dev libssl-dev pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
COPY schemas/ schemas/
COPY audio-stream-transcriber/Cargo.toml audio-stream-transcriber/Cargo.lock audio-stream-transcriber/build.rs audio-stream-transcriber/
COPY audio-stream-transcriber/src/ audio-stream-transcriber/src/
RUN cargo build --locked --release --manifest-path audio-stream-transcriber/Cargo.toml

FROM docker.io/debian:trixie-slim AS release
WORKDIR /app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates libglib2.0-0 libopus0 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/audio-stream-transcriber/target/release/audio-stream-transcriber ./
COPY audio-stream-transcriber/.env.development ./
COPY audio-stream-transcriber/.env.production ./
CMD ["./audio-stream-transcriber"]
