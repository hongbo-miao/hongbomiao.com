package main

import (
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/routes"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/utils"
	sharedUtils "github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/shared/utils"
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
		Str("ValkeyHost", config.ValkeyHost).
		Str("ValkeyPort", config.ValkeyPort).
		Str("ValkeyDB", config.ValkeyDB).
		Str("S3EndpointURL", config.S3EndpointURL).
		Str("S3Region", config.S3Region).
		Str("S3AccessKeyID", config.S3AccessKeyID).
		Str("OpenCensusAgentHost", config.OpenCensusAgentHost).
		Str("OpenCensusAgentPort", config.OpenCensusAgentPort).
		Str("EnableOpenTelemetryStdoutLog", config.EnableOpenTelemetryStdoutLog).
		Msg("main")

	valkeyDB, err := strconv.Atoi(config.ValkeyDB)
	if err != nil {
		log.Fatal().Err(err).Msg("strconv.Atoi")
	}
	valkeyClient, err := valkey.NewClient(valkey.ClientOption{
		InitAddress: []string{config.ValkeyHost + ":" + config.ValkeyPort},
		Password:    config.ValkeyPassword,
		SelectDB:    valkeyDB,
	})
	if err != nil {
		log.Fatal().Err(err).Msg("valkey.NewClient")
	}
	defer valkeyClient.Close()

	sharedUtils.InitOpenCensusTracer(config.OpenCensusAgentHost, config.OpenCensusAgentPort, "graphql_server")

	s3Client := s3.New(s3.Options{
		BaseEndpoint: aws.String(config.S3EndpointURL),
		Credentials:  credentials.NewStaticCredentialsProvider(config.S3AccessKeyID, config.S3SecretAccessKey, ""),
		Region:       config.S3Region,
		UsePathStyle: true,
	})

	r := routes.SetupRouter(config.AppEnv, valkeyClient, s3Client)
	_ = http.ListenAndServe(
		":"+config.Port,
		&ochttp.Handler{
			Handler: r,
		},
	)
}
