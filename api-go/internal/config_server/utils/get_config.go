package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	AppEnv                string
	Port                  string
	ShouldEnableServerTLS string
	ConfigServerCertPath  string
	ConfigServerKeyPath   string
	OPALAuthPublicKeyPath string
	PostgresHost          string
	PostgresPort          string
	PostgresDB            string
	PostgresUser          string
	PostgresPassword      string
}

func GetConfig() *Config {
	path := "config/config_server/"

	appEnv := os.Getenv("APP_ENV")
	if appEnv == "" {
		appEnv = "development"
	}

	_ = godotenv.Load(path + ".env." + appEnv + ".local")
	_ = godotenv.Load(path + ".env." + appEnv)

	return &Config{
		AppEnv:                appEnv,
		Port:                  os.Getenv("PORT"),
		ShouldEnableServerTLS: os.Getenv("SHOULD_ENABLE_SERVER_TLS"),
		ConfigServerCertPath:  os.Getenv("SERVER_CERT_PATH"),
		ConfigServerKeyPath:   os.Getenv("SERVER_KEY_PATH"),
		OPALAuthPublicKeyPath: os.Getenv("OPAL_AUTH_PUBLIC_KEY_PATH"),
		PostgresHost:          os.Getenv("POSTGRES_HOST"),
		PostgresPort:          os.Getenv("POSTGRES_PORT"),
		PostgresDB:            os.Getenv("POSTGRES_DB"),
		PostgresUser:          os.Getenv("POSTGRES_USER"),
		PostgresPassword:      os.Getenv("POSTGRES_PASSWORD"),
	}
}
