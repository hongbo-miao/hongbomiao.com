package main

import (
	"context"
	"fmt"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/api/api_server/proto/greet/v1"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/grpc_server/utils"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ocgrpc"
	"google.golang.org/grpc"
	"net"
)

type server struct{}

func (*server) Greet(ctx context.Context, req *v1.GreetRequest) (*v1.GreetResponse, error) {
	fmt.Printf("Greet function was invoked with %v\n", req)
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
		Str("appEnv", config.AppEnv).
		Str("port", config.Port).
		Str("openCensusAgentHost", config.OpenCensusAgentHost).
		Str("openCensusAgentPort", config.OpenCensusAgentPort).
		Msg("main")

	sharedUtils.InitOpenCensusTracer(config.OpenCensusAgentHost, config.OpenCensusAgentPort, "grpc_server")

	lis, err := net.Listen("tcp", ":"+config.Port)
	if err != nil {
		log.Error().Err(err).Msg("net.Listen")
	}

	s := grpc.NewServer(grpc.StatsHandler(&ocgrpc.ServerHandler{}))
	v1.RegisterGreetServiceServer(s, &server{})

	if err := s.Serve(lis); err != nil {
		log.Error().Err(err).Msg("s.Serve")
	}
}
