package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	Port                         string
	Env                          string
	GRPCHost                     string
	GRPCPort                     string
	OPAHost                      string
	OPAPort                      string
	DgraphHost                   string
	DgraphGRPCPort               string
	OpenCensusAgentHost          string
	OpenCensusAgentPort          string
	JaegerURL                    string
	JWTSecret                    string
	EnableOpenTelemetryStdoutLog string

	DecisionLogReceiverPort string
}

func GetConfig() *Config {
	env := os.Getenv("APP_ENV")
	if env == "" {
		env = "development"
	}

	_ = godotenv.Load(".env." + env + ".local")
	_ = godotenv.Load(".env." + env)
	_ = godotenv.Load() // .env

	return &Config{
		Env:                          env,
		Port:                         os.Getenv("PORT"),
		GRPCHost:                     os.Getenv("GRPC_HOST"),
		GRPCPort:                     os.Getenv("GRPC_PORT"),
		OPAHost:                      os.Getenv("OPA_HOST"),
		OPAPort:                      os.Getenv("OPA_PORT"),
		DgraphHost:                   os.Getenv("DGRAPH_HOST"),
		DgraphGRPCPort:               os.Getenv("DGRAPH_GRPC_PORT"),
		OpenCensusAgentHost:          os.Getenv("OPEN_CENSUS_AGENT_HOST"),
		OpenCensusAgentPort:          os.Getenv("OPEN_CENSUS_AGENT_PORT"),
		JaegerURL:                    os.Getenv("JAEGER_URL"),
		JWTSecret:                    os.Getenv("JWT_SECRET"),
		EnableOpenTelemetryStdoutLog: os.Getenv("ENABLE_OPEN_TELEMETRY_STDOUT_LOG"),

		DecisionLogReceiverPort: os.Getenv("DECISION_LOG_RECEIVER_PORT"),
	}
}
