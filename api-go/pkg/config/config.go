package config

import (
	"go.uber.org/zap"
)

type AppConfig struct {
	Logger *zap.Logger
}
