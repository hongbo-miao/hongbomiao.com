FROM docker.io/golang:1.24.5-alpine AS builder
WORKDIR /usr/src/app

COPY ["api-go/go.mod", "api-go/go.sum", "./"]
RUN go mod download

COPY api-go ./
RUN go build -o ./out/cmd/grpc_server/ ./cmd/grpc_server/main.go


FROM docker.io/alpine:3.22.1 AS release
WORKDIR /usr/src/app
ENV GIN_MODE=release

COPY --from=builder /usr/src/app/out/cmd/grpc_server/main ./main

EXPOSE 45237
CMD ["./main"]
