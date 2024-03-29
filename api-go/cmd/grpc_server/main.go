package main

import (
	"context"
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
	"github.com/hongbo-miao/hongbomiao.com/api-go/api/graphql_server/proto/greet/v1"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/grpc_server/utils"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"go.elastic.co/apm/module/apmgrpc/v2"
	"google.golang.org/grpc"
	"net"
	"os"
)

type server struct{}

func (*server) Greet(ctx context.Context, req *v1.GreetRequest) (*v1.GreetResponse, error) {
	firstName := req.GetFirstName()
	result := "Hello " + firstName
	res := &v1.GreetResponse{
		Result: result,
	}
	return res, nil
}

func main() {
	config := utils.GetConfig()
	log.Info().
		Str("AppEnv", config.AppEnv).
		Str("Port", config.Port).
		Str("OpenCensusAgentHost", config.OpenCensusAgentHost).
		Str("OpenCensusAgentPort", config.OpenCensusAgentPort).
		Msg("main")

	lis, err := net.Listen("tcp", ":"+config.Port)
	if err != nil {
		log.Error().Err(err).Msg("net.Listen")
	}

	logger := zerolog.New(os.Stderr)

	s := grpc.NewServer(
		grpc.ChainUnaryInterceptor(
			apmgrpc.NewUnaryServerInterceptor(apmgrpc.WithRecovery()),
			logging.UnaryServerInterceptor(utils.InterceptLogger(logger)),
		),
		grpc.ChainStreamInterceptor(
			apmgrpc.NewStreamServerInterceptor(apmgrpc.WithRecovery()),
			logging.StreamServerInterceptor(utils.InterceptLogger(logger)),
		),
	)
	v1.RegisterGreetServiceServer(s, &server{})

	if err := s.Serve(lis); err != nil {
		log.Error().Err(err).Msg("s.Serve")
	}
}
