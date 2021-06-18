package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	CORSAllowOrigins *map[string]bool
	Port             string
	Env              string
	GRPCHost         string
	GRPCPort         string
}

func getCORSAllowOrigins(env string, webHost string) *map[string]bool {
	var devCORSAllowOrigins = map[string]bool{
		"electron://altair": true,
		webHost:             true,
	}
	var prodCORSAllowOrigins = map[string]bool{
		"electron://altair":          true,
		"https://www.hongbomiao.com": true,
		webHost:                      true,
	}

	if env == "production" {
		return &prodCORSAllowOrigins
	}
	return &devCORSAllowOrigins
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
		CORSAllowOrigins: getCORSAllowOrigins(env, os.Getenv("WEB_HOST")),
		Env:              env,
		Port:             os.Getenv("PORT"),
		GRPCHost:         os.Getenv("GRPC_HOST"),
		GRPCPort:         os.Getenv("GRPC_PORT"),
	}
}
