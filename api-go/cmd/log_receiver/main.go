package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/log_receiver/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/log_receiver/utils"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/gin-contrib/gzip"
	"github.com/gin-contrib/logger"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

func main() {
	sharedUtils.InitLogger()
	var config = utils.GetConfig()
	log.Info().
		Str("AppEnv", config.AppEnv).
		Str("Port", config.Port).
		Msg("main")

	r := gin.Default()
	r.Use(logger.SetLogger())
	r.Use(gzip.Gzip(gzip.DefaultCompression, gzip.WithDecompressFn(gzip.DefaultDecompressHandle)))
	r.POST("/logs", controllers.Logs)
	err := r.Run(":" + config.Port)
	if err != nil {
		log.Error().Err(err).Msg("r.Run")
	}
}
