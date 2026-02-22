package utils

import (
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	AppEnv              string
	Port                string
	OpenCensusAgentHost string
	OpenCensusAgentPort string
}

func GetConfig() *Config {
	path := "config/grpc_server/"

	appEnv := os.Getenv("ENVIRONMENT")
	if appEnv == "" {
		appEnv = "development"
	}

	_ = godotenv.Load(path + ".env." + appEnv + ".local")
	_ = godotenv.Load(path + ".env." + appEnv)

	return &Config{
		AppEnv:              appEnv,
		Port:                os.Getenv("PORT"),
		OpenCensusAgentHost: os.Getenv("OPEN_CENSUS_AGENT_HOST"),
		OpenCensusAgentPort: os.Getenv("OPEN_CENSUS_AGENT_PORT"),
	}
}
