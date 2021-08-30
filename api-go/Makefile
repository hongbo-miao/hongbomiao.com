run-graphql-server:
	go run cmd/graphql_server/main.go
run-grpc-server:
	go run cmd/grpc_server/main.go
run-log-receiver:
	go run cmd/log_receiver/main.go

go-tidy:
	go mod tidy
go-test:
	go test ./...
go-fmt:
	gofmt .
go-check:
	gofmt -l .
go-lint:
	golangci-lint run ./...

proto-generate:
	protoc api/graphql_server/proto/greet/v1/greet.proto --go_out=plugins=grpc:.
	protoc api/graphql_server/proto/torchserve/v1/inference.proto --go_out=plugins=grpc:.
	protoc api/graphql_server/proto/torchserve/v1/management.proto --go_out=plugins=grpc:.
proto-lint:
	buf lint

opa-test:
	opa test internal/graphql_server/policies/*.rego -v
opa-test-coverage:
	opa test internal/graphql_server/policies/*.rego --coverage --format=json
opa-fmt:
	opa fmt internal/graphql_server/policies/*.rego -w
opa-check:
	opa check internal/graphql_server/policies/*.rego

gzip-call:
	echo '{"foo": "bar"}' | gzip | curl -v -i --data-binary @- -H "Content-Encoding: gzip" http://localhost:41797/logs
