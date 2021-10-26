package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	AppEnv                string
	Port                  string
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
		OPALAuthPublicKeyPath: os.Getenv("OPAL_AUTH_PUBLIC_KEY_PATH"),
	}
}
