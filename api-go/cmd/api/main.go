package main

import (
	"contrib.go.opencensus.io/exporter/ocagent"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/routes"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ochttp"
	"go.opencensus.io/trace"
	"net/http"
	"time"
)

func main() {
	utils.InitLogger()
	var config = utils.GetConfig()
	log.Info().Str("env", config.Env).Str("port", config.Port).Msg("main")

	oce, err := ocagent.NewExporter(
		ocagent.WithInsecure(),
		ocagent.WithReconnectionPeriod(5*time.Second),
		ocagent.WithAddress(config.OpenCensusAgentHost+":"+config.OpenCensusAgentPort),
		ocagent.WithServiceName("api-go"))
	if err != nil {
		log.Error().Err(err).Msg("ocagent.NewExporter")
	}
	trace.RegisterExporter(oce)

	r := routes.SetupRouter()
	_ = http.ListenAndServe(
		":"+config.Port,
		&ochttp.Handler{
			Handler: r,
		},
	)
}
