module github.com/Hongbo-Miao/hongbomiao.com/api-go

go 1.16

require (
	contrib.go.opencensus.io/exporter/ocagent v0.7.0
	github.com/buger/jsonparser v1.1.1
	github.com/dgraph-io/dgo/v200 v200.0.0-20210401091508-95bfd74de60e
	github.com/gin-contrib/cors v1.3.1
	github.com/gin-contrib/gzip v0.0.3
	github.com/gin-gonic/gin v1.7.2
	github.com/go-playground/validator/v10 v10.8.0 // indirect
	github.com/go-resty/resty/v2 v2.6.0
	github.com/golang-jwt/jwt v3.2.1+incompatible
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da // indirect
	github.com/graphql-go/graphql v0.7.9
	github.com/graphql-go/handler v0.2.3
	github.com/joho/godotenv v1.3.0
	github.com/json-iterator/go v1.1.11 // indirect
	github.com/mattn/go-isatty v0.0.13 // indirect
	github.com/open-policy-agent/opa v0.31.0
	github.com/rcrowley/go-metrics v0.0.0-20201227073835-cf1acfcdf475 // indirect
	github.com/rs/zerolog v1.23.0
	github.com/stretchr/testify v1.7.0
	github.com/ugorji/go v1.2.6 // indirect
	go.opencensus.io v0.23.0
	go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin v0.21.0
	go.opentelemetry.io/otel v1.0.0-RC2
	go.opentelemetry.io/otel/exporters/jaeger v1.0.0-RC1
	go.opentelemetry.io/otel/exporters/stdout/stdouttrace v1.0.0-RC1
	go.opentelemetry.io/otel/sdk v1.0.0-RC2
	golang.org/x/net v0.0.0-20210716203947-853a461950ff // indirect
	google.golang.org/genproto v0.0.0-20210722135532-667f2b7c528f // indirect
	google.golang.org/grpc v1.39.0
	google.golang.org/protobuf v1.27.1
	gopkg.in/yaml.v3 v3.0.0-20210107192922-496545a6307b // indirect
)
