package main

import (
	"context"
	"contrib.go.opencensus.io/exporter/ocagent"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/routes"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ochttp"
	opencensustrace "go.opencensus.io/trace"
	"net/http"
	"time"
)

func main() {
	utils.InitLogger()
	var config = utils.GetConfig()
	log.Info().
		Str("env", config.Env).
		Str("port", config.Port).
		Str("openCensusAgentHost", config.OpenCensusAgentHost).
		Str("openCensusAgentPort", config.OpenCensusAgentPort).
		Str("JaegerURL", config.JaegerURL).
		Msg("main")

	tp := utils.InitTracer(config.EnableOpenTelemetryStdoutLog, config.JaegerURL)
	defer func() {
		if err := tp.Shutdown(context.Background()); err != nil {
			log.Error().Err(err).Msg("tp.Shutdown")
		}
	}()

	oce, err := ocagent.NewExporter(
		ocagent.WithInsecure(),
		ocagent.WithReconnectionPeriod(5*time.Second),
		ocagent.WithAddress(config.OpenCensusAgentHost+":"+config.OpenCensusAgentPort),
		ocagent.WithServiceName("api-server"))
	if err != nil {
		log.Error().Err(err).Msg("ocagent.NewExporter")
	}
	opencensustrace.RegisterExporter(oce)

	r := routes.SetupRouter()
	_ = http.ListenAndServe(
		":"+config.Port,
		&ochttp.Handler{
			Handler: r,
		},
	)
}
