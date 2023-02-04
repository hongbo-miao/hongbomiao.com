package main

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/routes"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ochttp"
	"net/http"
	"strconv"
)

func main() {
	sharedUtils.InitLogger()
	config := utils.GetConfig()
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
		Str("MinIOEndpoint", config.MinIOEndpoint).
		Str("MinIOAccessKeyID", config.MinIOAccessKeyID).
		Str("OpenCensusAgentHost", config.OpenCensusAgentHost).
		Str("OpenCensusAgentPort", config.OpenCensusAgentPort).
		Str("JaegerURL", config.JaegerURL).
		Str("EnableOpenTelemetryStdoutLog", config.EnableOpenTelemetryStdoutLog).
		Msg("main")

	redisDB, err := strconv.Atoi(config.RedisDB)
	if err != nil {
		log.Error().Err(err).Msg("strconv.Atoi")
	}
	rdb := redis.NewClient(&redis.Options{
		Addr:     config.RedisHost + ":" + config.RedisPort,
		Password: config.RedisPassword,
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

	sharedUtils.InitOpenCensusTracer(config.OpenCensusAgentHost, config.OpenCensusAgentPort, "graphql_server")

	minioClient, err := minio.New(config.MinIOEndpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(config.MinIOAccessKeyID, config.MinIOSecretAccessKey, ""),
		Secure: true,
	})
	if err != nil {
		log.Error().Err(err).Msg("minio.New")
	}

	r := routes.SetupRouter(config.AppEnv, rdb, minioClient)
	_ = http.ListenAndServe(
		":"+config.Port,
		&ochttp.Handler{
			Handler: r,
		},
	)
}
