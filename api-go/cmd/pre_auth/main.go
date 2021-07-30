package main

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/pre_auth/routes"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/pre_auth/utils"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ochttp"
	"net/http"
)

func main() {
	sharedUtils.InitLogger()
	var config = utils.GetConfig()
	log.Info().
		Str("appEnv", config.AppEnv).
		Str("port", config.Port).
		Str("openCensusAgentHost", config.OpenCensusAgentHost).
		Str("openCensusAgentPort", config.OpenCensusAgentPort).
		Str("JaegerURL", config.JaegerURL).
		Msg("main")

	tp, err := utils.InitTracer(config.EnableOpenTelemetryStdoutLog, config.JaegerURL)
	if err != nil {
		log.Error().Err(err).Msg("InitTracer")
	}
	defer func() {
		if err := tp.Shutdown(context.Background()); err != nil {
			log.Error().Err(err).Msg("tp.Shutdown")
		}
	}()

	sharedUtils.InitOpenCensusTracer(config.OpenCensusAgentHost, config.OpenCensusAgentPort, "pre_auth")

	r := routes.SetupRouter(config.AppEnv)
	_ = http.ListenAndServe(
		":"+config.Port,
		&ochttp.Handler{
			Handler: r,
		},
	)
}
