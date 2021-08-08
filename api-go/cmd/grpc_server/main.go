package main

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/api/api_server/proto/greet/v1"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/grpc_server/utils"
	grpczerolog "github.com/grpc-ecosystem/go-grpc-middleware/providers/zerolog/v2"
	middleware "github.com/grpc-ecosystem/go-grpc-middleware/v2"
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/tags"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"go.elastic.co/apm/module/apmgrpc"
	"google.golang.org/grpc"
	"net"
	"os"
)

type server struct{}

func (*server) Greet(ctx context.Context, req *v1.GreetRequest) (*v1.GreetResponse, error) {
	firstName := req.GetGreeting().GetFirstName()
	result := "Hello " + firstName
	res := &v1.GreetResponse{
		Result: result,
	}
	return res, nil
}

func main() {
	var config = utils.GetConfig()
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
		middleware.WithUnaryServerChain(
			apmgrpc.NewUnaryServerInterceptor(apmgrpc.WithRecovery()),
			tags.UnaryServerInterceptor(),
			logging.UnaryServerInterceptor(grpczerolog.InterceptorLogger(logger)),
		),
		middleware.WithStreamServerChain(
			apmgrpc.NewStreamServerInterceptor(apmgrpc.WithRecovery()),
			tags.StreamServerInterceptor(),
			logging.StreamServerInterceptor(grpczerolog.InterceptorLogger(logger)),
		),
	)
	v1.RegisterGreetServiceServer(s, &server{})

	if err := s.Serve(lis); err != nil {
		log.Error().Err(err).Msg("s.Serve")
	}
}
