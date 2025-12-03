FROM docker.io/golang:1.25.5-alpine AS builder
WORKDIR /usr/src/app

COPY ["api/api-go/go.mod", "api/api-go/go.sum", "./"]
RUN go mod download

COPY api/api-go ./
RUN go build -o ./out/cmd/grpc_server/ ./cmd/grpc_server/main.go


FROM docker.io/alpine:3.22.2 AS release
WORKDIR /usr/src/app
ENV GIN_MODE=release

COPY --from=builder /usr/src/app/out/cmd/grpc_server/main ./main

EXPOSE 45237
CMD ["./main"]
