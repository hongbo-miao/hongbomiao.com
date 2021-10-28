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
	}
}
