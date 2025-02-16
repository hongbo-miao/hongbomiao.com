FROM docker.io/golang:1.24.0-alpine AS builder
WORKDIR /usr/src/app

COPY ["api-go/go.mod", "api-go/go.sum", "./"]
RUN go mod download

COPY api-go ./
RUN go build -o ./out/cmd/config_loader/ ./cmd/config_loader/main.go


FROM docker.io/alpine:3.21.3 AS release
WORKDIR /usr/src/app
ENV GIN_MODE=release

COPY --from=builder /usr/src/app/out/cmd/config_loader/main ./main

EXPOSE 26660
CMD ["./main"]
