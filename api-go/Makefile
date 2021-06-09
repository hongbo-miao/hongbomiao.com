run:
	go run cmd/api/main.go

format:
	go fmt ./...

tidy:
	go mod tidy

test:
	opa test **/*.rego -v

test-coverage:
	opa test **/*.rego --coverage --format=json
