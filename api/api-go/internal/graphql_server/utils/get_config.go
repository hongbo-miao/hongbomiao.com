package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	AppEnv                       string
	Port                         string
	GRPCServerHost               string
	GRPCServerPort               string
	OPAHost                      string
	OPAPort                      string
	DgraphHost                   string
	DgraphGRPCPort               string
	RedisHost                    string
	RedisPort                    string
	RedisDB                      string
	RedisPassword                string
	MinIOEndpoint                string
	MinIOAccessKeyID             string
	MinIOSecretAccessKey         string
	TorchServeGRPCHost           string
	TorchServeGRPCPort           string
	OpenCensusAgentHost          string
	OpenCensusAgentPort          string
	JWTSecret                    string
	EnableOpenTelemetryStdoutLog string
}

func GetConfig() *Config {
	path := "config/graphql_server/"

	appEnv := os.Getenv("APP_ENV")
	if appEnv == "" {
		appEnv = "development"
	}

	_ = godotenv.Load(path + ".env." + appEnv + ".local")
	_ = godotenv.Load(path + ".env." + appEnv)

	return &Config{
		AppEnv:                       appEnv,
		Port:                         os.Getenv("PORT"),
		GRPCServerHost:               os.Getenv("GRPC_SERVER_HOST"),
		GRPCServerPort:               os.Getenv("GRPC_SERVER_PORT"),
		OPAHost:                      os.Getenv("OPA_HOST"),
		OPAPort:                      os.Getenv("OPA_PORT"),
		DgraphHost:                   os.Getenv("DGRAPH_HOST"),
		DgraphGRPCPort:               os.Getenv("DGRAPH_GRPC_PORT"),
		RedisHost:                    os.Getenv("REDIS_HOST"),
		RedisPort:                    os.Getenv("REDIS_PORT"),
		RedisDB:                      os.Getenv("REDIS_DB"),
		RedisPassword:                os.Getenv("REDIS_PASSWORD"),
		MinIOEndpoint:                os.Getenv("MINIO_ENDPOINT"),
		MinIOAccessKeyID:             os.Getenv("MINIO_ACCESS_KEY_ID"),
		MinIOSecretAccessKey:         os.Getenv("MINIO_SECRET_ACCESS_KEY"),
		TorchServeGRPCHost:           os.Getenv("TORCH_SERVE_GRPC_HOST"),
		TorchServeGRPCPort:           os.Getenv("TORCH_SERVE_GRPC_PORT"),
		OpenCensusAgentHost:          os.Getenv("OPEN_CENSUS_AGENT_HOST"),
		OpenCensusAgentPort:          os.Getenv("OPEN_CENSUS_AGENT_PORT"),
		JWTSecret:                    os.Getenv("JWT_SECRET"),
		EnableOpenTelemetryStdoutLog: os.Getenv("ENABLE_OPEN_TELEMETRY_STDOUT_LOG"),
	}
}
