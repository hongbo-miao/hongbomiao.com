FROM golang:1.17.0-alpine AS builder
WORKDIR /usr/src/app

COPY ["api-go/go.mod", "api-go/go.sum", "./"]
RUN go mod download

COPY api-go ./
RUN go build -o ./out/cmd/pre_auth/ ./cmd/pre_auth/main.go


FROM alpine:3.14.1 AS release
WORKDIR /usr/src/app
ENV GIN_MODE=release

COPY --from=builder /usr/src/app/out/cmd/pre_auth/main ./main

EXPOSE 29930
CMD ["./main"]
