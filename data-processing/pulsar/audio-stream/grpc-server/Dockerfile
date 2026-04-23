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
COPY grpc-server/Cargo.toml grpc-server/Cargo.lock grpc-server/build.rs grpc-server/
COPY grpc-server/src/ grpc-server/src/
RUN cargo build --locked --release --manifest-path grpc-server/Cargo.toml

FROM docker.io/debian:trixie-slim AS release
WORKDIR /app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    libopus0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/grpc-server/target/release/grpc-server ./
COPY grpc-server/.env.development ./
COPY grpc-server/.env.production ./
CMD ["./grpc-server"]
