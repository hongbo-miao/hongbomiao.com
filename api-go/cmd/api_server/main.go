package main

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/routes"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/utils"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/go-redis/redis/v8"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ochttp"
	"net/http"
	"strconv"
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
		Str("RedisHost", config.RedisHost).
		Str("RedisPort", config.RedisPort).
		Str("RedisDB", config.RedisDB).
		Str("OpenCensusAgentHost", config.OpenCensusAgentHost).
		Str("OpenCensusAgentPort", config.OpenCensusAgentPort).
		Str("JaegerURL", config.JaegerURL).
		Msg("main")

	redisDB, err := strconv.Atoi(config.RedisDB)
	if err != nil {
		log.Error().Err(err).Msg("strconv.Atoi")
	}
	rdb := redis.NewClient(&redis.Options{
		Addr:     config.RedisHost + ":" + config.RedisPort,
		Password: "",
		DB:       redisDB,
	})

	defer func(rdb *redis.Client) {
		err := rdb.Close()
		if err != nil {
			log.Error().Err(err).Msg("rdb.Close")
		}
	}(rdb)

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

	r := routes.SetupRouter(config.AppEnv, rdb)
	_ = http.ListenAndServe(
		":"+config.Port,
		&ochttp.Handler{
			Handler: r,
		},
	)
}
