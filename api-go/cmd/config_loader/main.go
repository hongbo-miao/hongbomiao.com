package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/config_loader/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/config_loader/utils"
	sharedControllers "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/controllers"
	sharedHandlers "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/handlers"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
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
		Str("ShouldEnableServerTLS", config.ShouldEnableServerTLS).
		Str("ConfigLoaderCertPath", config.ConfigLoaderCertPath).
		Str("ConfigLoaderKeyPath", config.ConfigLoaderKeyPath).
		Str("OPALAuthPublicKeyPath", config.OPALAuthPublicKeyPath).
		Str("PostgresHost", config.PostgresHost).
		Str("PostgresPort", config.PostgresPort).
		Str("PostgresDB", config.PostgresDB).
		Str("PostgresUser", config.PostgresUser).
		Msg("main")

	pg := utils.InitPostgres(
		config.PostgresHost,
		config.PostgresPort,
		config.PostgresDB,
		config.PostgresUser,
		config.PostgresPassword)
	defer pg.Close()

	r := gin.New()
	r.Use(apmgin.Middleware(r))
	r.Use(logger.SetLogger())
	r.GET("/", sharedControllers.Health)
	r.GET("/metrics", sharedHandlers.PrometheusHandler())
	r.GET("/config", controllers.Config(pg))
	r.POST("/opa-db-cdc", controllers.OPADBCDC(pg))
	if config.ShouldEnableServerTLS == "true" {
		err := r.RunTLS(":"+config.Port, config.ConfigLoaderCertPath, config.ConfigLoaderKeyPath)
		if err != nil {
			log.Error().Err(err).Msg("r.RunTLS")
		}
	} else {
		err := r.Run(":" + config.Port)
		if err != nil {
			log.Error().Err(err).Msg("r.Run")
		}
	}
}
