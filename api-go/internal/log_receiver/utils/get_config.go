package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	AppEnv                string
	Port                  string
	ElasticAPMServiceName string
	ElasticAPMServerURL   string
}

func GetConfig() *Config {
	path := "config/log_receiver/"

	appEnv := os.Getenv("APP_ENV")
	if appEnv == "" {
		appEnv = "development"
	}

	_ = godotenv.Load(path + ".env." + appEnv + ".local")
	_ = godotenv.Load(path + ".env." + appEnv)

	return &Config{
		AppEnv:                appEnv,
		Port:                  os.Getenv("PORT"),
		ElasticAPMServiceName: os.Getenv("ELASTIC_APM_SERVICE_NAME"),
		ElasticAPMServerURL:   os.Getenv("ELASTIC_APM_SERVER_URL"),
	}
}
