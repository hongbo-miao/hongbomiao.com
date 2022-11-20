package utils

import (
	"context"
	greet "github.com/Hongbo-Miao/hongbomiao.com/api-go/api/graphql_server/proto/greet/v1"
	"github.com/rs/zerolog/log"
	"go.elastic.co/apm/module/apmgrpc/v2"
	"go.opencensus.io/plugin/ocgrpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Greeting struct {
	Content string `json:"content"`
}

func GetGreeting(firstName string, lastName string) (*Greeting, error) {
	req := &greet.GreetRequest{
		FirstName: firstName,
		LastName:  lastName,
	}

	config := GetConfig()
	conn, err := grpc.Dial(
		config.GRPCServerHost+":"+config.GRPCServerPort,
		grpc.WithUnaryInterceptor(apmgrpc.NewUnaryClientInterceptor()),
		grpc.WithStatsHandler(new(ocgrpc.ClientHandler)),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Error().Err(err).Msg("grpc.Dial")
		return nil, err
	}
	defer func(conn *grpc.ClientConn) {
		err := conn.Close()
		if err != nil {
			log.Error().Err(err).Msg("conn.Close")
		}
	}(conn)

	c := greet.NewGreetServiceClient(conn)
	res, err := c.Greet(context.Background(), req)
	if err != nil {
		log.Error().Err(err).Msg("c.Greet")
		return nil, err
	}

	return &Greeting{
		Content: res.Result,
	}, nil
}
