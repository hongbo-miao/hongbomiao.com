package utils

import (
	"contrib.go.opencensus.io/exporter/ocagent"
	"github.com/rs/zerolog/log"
	opencensustrace "go.opencensus.io/trace"
	"time"
)

func InitOpenCensusTracer(openCensusAgentHost string, openCensusAgentPort string, serviceName string) {
	oce, err := ocagent.NewExporter(
		ocagent.WithInsecure(),
		ocagent.WithReconnectionPeriod(5*time.Second),
		ocagent.WithAddress(openCensusAgentHost+":"+openCensusAgentPort),
		ocagent.WithServiceName(serviceName))
	if err != nil {
		log.Error().Err(err).Msg("ocagent.NewExporter")
		return
	}
	opencensustrace.RegisterExporter(oce)
}
