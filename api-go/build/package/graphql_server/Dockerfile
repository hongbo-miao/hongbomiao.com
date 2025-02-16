FROM docker.io/golang:1.24.0-alpine AS builder
WORKDIR /usr/src/app

COPY ["api-go/go.mod", "api-go/go.sum", "./"]
RUN go mod download

COPY api-go ./
RUN go build -o ./out/cmd/graphql_server/ ./cmd/graphql_server/main.go


FROM docker.io/alpine:3.21.3 AS release
WORKDIR /usr/src/app
ENV GIN_MODE=release

COPY --from=builder /usr/src/app/out/cmd/graphql_server/main ./main

EXPOSE 31800
CMD ["./main"]
