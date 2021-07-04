package main

import (
	"context"
	"contrib.go.opencensus.io/exporter/ocagent"
	"fmt"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/api/proto/greet/v1"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ocgrpc"
	"go.opencensus.io/trace"
	"google.golang.org/grpc"
	"net"
	"time"
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
		Str("env", config.Env).
		Str("grpcPort", config.GRPCPort).
		Str("openCensusAgentHost", config.OpenCensusAgentHost).
		Str("openCensusAgentPort", config.OpenCensusAgentPort).
		Msg("main")

	oce, err := ocagent.NewExporter(
		ocagent.WithInsecure(),
		ocagent.WithReconnectionPeriod(5*time.Second),
		ocagent.WithAddress(config.OpenCensusAgentHost+":"+config.OpenCensusAgentPort),
		ocagent.WithServiceName("voting"))
	if err != nil {
		log.Error().Err(err).Msg("ocagent.NewExporter")
	}
	trace.RegisterExporter(oce)

	lis, err := net.Listen("tcp", ":"+config.GRPCPort)
	if err != nil {
		log.Error().Err(err).Msg("net.Listen")
	}

	s := grpc.NewServer(grpc.StatsHandler(&ocgrpc.ServerHandler{}))
	v1.RegisterGreetServiceServer(s, &server{})

	if err := s.Serve(lis); err != nil {
		log.Error().Err(err).Msg("s.Serve")
	}
}
