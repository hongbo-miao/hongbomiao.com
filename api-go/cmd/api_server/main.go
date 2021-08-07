package main

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/routes"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/utils"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ochttp"
	"net/http"
)

func main() {
	sharedUtils.InitLogger()
	var config = utils.GetConfig()
	log.Info().
		Str("AppEnv", config.AppEnv).
		Str("Port", config.Port).
		Str("GRPCServerHost", config.GRPCServerHost).
		Str("GRPCServerPort", config.GRPCServerPort).
		Str("OPAHost", config.OPAHost).
		Str("OPAPort", config.OPAPort).
		Str("DgraphHost", config.DgraphHost).
		Str("DgraphGRPCPort", config.DgraphGRPCPort).
		Str("ElasticAPMServiceName", config.ElasticAPMServiceName).
		Str("ElasticAPMServerURL", config.ElasticAPMServerURL).
		Str("OpenCensusAgentHost", config.OpenCensusAgentHost).
		Str("OpenCensusAgentPort", config.OpenCensusAgentPort).
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

	sharedUtils.InitOpenCensusTracer(config.OpenCensusAgentHost, config.OpenCensusAgentPort, "api_server")

	r := routes.SetupRouter(config.AppEnv)
	_ = http.ListenAndServe(
		":"+config.Port,
		&ochttp.Handler{
			Handler: r,
		},
	)
}
