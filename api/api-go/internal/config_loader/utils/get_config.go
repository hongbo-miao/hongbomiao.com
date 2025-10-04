package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	AppEnv                string
	Port                  string
	ShouldEnableServerTLS string
	ConfigLoaderCertPath  string
	ConfigLoaderKeyPath   string
	OPALAuthPublicKeyPath string
	OPALServerHost        string
	OPALServerPort        string
	OPALClientToken       string
	PostgresHost          string
	PostgresPort          string
	PostgresDB            string
	PostgresUser          string
	PostgresPassword      string
}

func GetConfig() *Config {
	path := "config/config_loader/"

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
		ConfigLoaderCertPath:  os.Getenv("SERVER_CERT_PATH"),
		ConfigLoaderKeyPath:   os.Getenv("SERVER_KEY_PATH"),
		OPALAuthPublicKeyPath: os.Getenv("OPAL_AUTH_PUBLIC_KEY_PATH"),
		OPALServerHost:        os.Getenv("OPAL_SERVER_HOST"),
		OPALServerPort:        os.Getenv("OPAL_SERVER_PORT"),
		OPALClientToken:       os.Getenv("OPAL_CLIENT_TOKEN"),
		PostgresHost:          os.Getenv("POSTGRES_HOST"),
		PostgresPort:          os.Getenv("POSTGRES_PORT"),
		PostgresDB:            os.Getenv("POSTGRES_DB"),
		PostgresUser:          os.Getenv("POSTGRES_USER"),
		PostgresPassword:      os.Getenv("POSTGRES_PASSWORD"),
	}
}
