package utils

import (
	"os"

	"github.com/joho/godotenv"
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
	ValkeyHost                   string
	ValkeyPort                   string
	ValkeyDB                     string
	ValkeyPassword               string
	S3EndpointURL                string
	S3Region                     string
	S3AccessKeyID                string
	S3SecretAccessKey            string
	TorchServeGRPCHost           string
	TorchServeGRPCPort           string
	OpenCensusAgentHost          string
	OpenCensusAgentPort          string
	JWTSecret                    string
	EnableOpenTelemetryStdoutLog string
}

func GetConfig() *Config {
	path := "config/graphql_server/"

	appEnv := os.Getenv("ENVIRONMENT")
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
		ValkeyHost:                   os.Getenv("VALKEY_HOST"),
		ValkeyPort:                   os.Getenv("VALKEY_PORT"),
		ValkeyDB:                     os.Getenv("VALKEY_DB"),
		ValkeyPassword:               os.Getenv("VALKEY_PASSWORD"),
		S3EndpointURL:                os.Getenv("S3_ENDPOINT_URL"),
		S3Region:                     os.Getenv("S3_REGION"),
		S3AccessKeyID:                os.Getenv("S3_ACCESS_KEY_ID"),
		S3SecretAccessKey:            os.Getenv("S3_SECRET_ACCESS_KEY"),
		TorchServeGRPCHost:           os.Getenv("TORCH_SERVE_GRPC_HOST"),
		TorchServeGRPCPort:           os.Getenv("TORCH_SERVE_GRPC_PORT"),
		OpenCensusAgentHost:          os.Getenv("OPEN_CENSUS_AGENT_HOST"),
		OpenCensusAgentPort:          os.Getenv("OPEN_CENSUS_AGENT_PORT"),
		JWTSecret:                    os.Getenv("JWT_SECRET"),
		EnableOpenTelemetryStdoutLog: os.Getenv("ENABLE_OPEN_TELEMETRY_STDOUT_LOG"),
	}
}
