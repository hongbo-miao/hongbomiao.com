package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/decision_logger/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/decision_logger/utils"
	sharedControllers "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/controllers"
	sharedHandlers "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/handlers"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/gin-contrib/gzip"
	"github.com/gin-contrib/logger"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"go.elastic.co/apm/module/apmgin/v2"
)

func main() {
	sharedUtils.InitLogger()
	config := utils.GetConfig()
	log.Info().
		Str("AppEnv", config.AppEnv).
		Str("Port", config.Port).
		Msg("main")

	r := gin.New()
	r.Use(apmgin.Middleware(r))
	r.Use(logger.SetLogger())
	r.GET("/", sharedControllers.Health)
	r.GET("/metrics", sharedHandlers.PrometheusHandler())
	r.Use(gzip.Gzip(gzip.DefaultCompression, gzip.WithDecompressFn(gzip.DefaultDecompressHandle)))
	r.POST("/logs", controllers.Logs)
	err := r.Run(":" + config.Port)
	if err != nil {
		log.Error().Err(err).Msg("r.Run")
	}
}
