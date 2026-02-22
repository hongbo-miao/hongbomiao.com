package main

import (
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/routes"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/utils"
	sharedUtils "github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/shared/utils"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/rs/zerolog/log"
	"github.com/valkey-io/valkey-go"
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
		Str("EnableOpenTelemetryStdoutLog", config.EnableOpenTelemetryStdoutLog).
		Msg("main")

	redisDB, err := strconv.Atoi(config.RedisDB)
	if err != nil {
		log.Error().Err(err).Msg("strconv.Atoi")
	}
	valkeyClient, err := valkey.NewClient(valkey.ClientOption{
		InitAddress: []string{config.RedisHost + ":" + config.RedisPort},
		Password:    config.RedisPassword,
		SelectDB:    redisDB,
	})
	if err != nil {
		log.Fatal().Err(err).Msg("valkey.NewClient")
	}
	defer valkeyClient.Close()

	sharedUtils.InitOpenCensusTracer(config.OpenCensusAgentHost, config.OpenCensusAgentPort, "graphql_server")

	minioClient, err := minio.New(config.MinIOEndpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(config.MinIOAccessKeyID, config.MinIOSecretAccessKey, ""),
		Secure: true,
	})
	if err != nil {
		log.Error().Err(err).Msg("minio.New")
	}

	r := routes.SetupRouter(config.AppEnv, valkeyClient, minioClient)
	_ = http.ListenAndServe(
		":"+config.Port,
		&ochttp.Handler{
			Handler: r,
		},
	)
}
