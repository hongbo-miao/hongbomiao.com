module github.com/Hongbo-Miao/hongbomiao.com/api-go

go 1.16

require (
	contrib.go.opencensus.io/exporter/ocagent v0.7.0
	github.com/Masterminds/squirrel v1.5.3
	github.com/bep/debounce v1.2.1
	github.com/buger/jsonparser v1.1.1
	github.com/dgraph-io/dgo/v200 v200.0.0-20210401091508-95bfd74de60e
	github.com/elastic/go-sysinfo v1.7.0 // indirect
	github.com/elastic/go-windows v1.0.1 // indirect
	github.com/gin-contrib/cors v1.4.0
	github.com/gin-contrib/gzip v0.0.6
	github.com/gin-contrib/logger v0.2.2
	github.com/gin-gonic/gin v1.8.1
	github.com/go-redis/redis/v8 v8.11.5
	github.com/go-redis/redismock/v8 v8.0.6
	github.com/go-resty/resty/v2 v2.7.0
	github.com/golang-jwt/jwt/v4 v4.4.2
	github.com/google/uuid v1.3.0 // indirect
	github.com/graphql-go/graphql v0.8.0
	github.com/graphql-go/handler v0.2.3
	github.com/grpc-ecosystem/go-grpc-middleware/providers/zerolog/v2 v2.0.0-rc.2
	github.com/grpc-ecosystem/go-grpc-middleware/v2 v2.0.0-rc.2
	github.com/jackc/pgx/v4 v4.16.1
	github.com/joho/godotenv v1.4.0
	github.com/minio/md5-simd v1.1.2 // indirect
	github.com/minio/minio-go/v7 v7.0.34
	github.com/minio/sha256-simd v1.0.0 // indirect
	github.com/open-policy-agent/opa v0.43.0
	github.com/prometheus/client_golang v1.12.2
	github.com/prometheus/common v0.33.0 // indirect
	github.com/rcrowley/go-metrics v0.0.0-20201227073835-cf1acfcdf475 // indirect
	github.com/rs/zerolog v1.27.0
	github.com/stretchr/testify v1.8.0
	go.elastic.co/apm/module/apmgin v1.15.0
	go.elastic.co/apm/module/apmgrpc v1.15.0
	go.opencensus.io v0.23.0
	go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin v0.34.0
	go.opentelemetry.io/otel v1.9.0
	go.opentelemetry.io/otel/exporters/jaeger v1.9.0
	go.opentelemetry.io/otel/exporters/stdout/stdouttrace v1.9.0
	go.opentelemetry.io/otel/sdk v1.9.0
	go.uber.org/atomic v1.9.0
	google.golang.org/grpc v1.48.0
	google.golang.org/grpc/examples v0.0.0-20210806175644-574137db7de3 // indirect
	google.golang.org/protobuf v1.28.1
	howett.net/plist v0.0.0-20201203080718-1454fab16a06 // indirect
)
