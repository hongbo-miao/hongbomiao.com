package utils

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/api/api_server/proto/greet/v1"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/plugin/ocgrpc"
	"google.golang.org/grpc"
)

type Greeting struct {
	Content string `json:"content"`
}

func GetGreeting(firstName string, lastName string) (*Greeting, error) {
	req := &v1.GreetRequest{
		Greeting: &v1.Greeting{
			FirstName: firstName,
			LastName:  lastName,
		},
	}

	config := GetConfig()
	conn, err := grpc.Dial(config.GRPCServerHost+":"+config.GRPCServerPort, grpc.WithInsecure(),
		grpc.WithStatsHandler(new(ocgrpc.ClientHandler)))
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

	c := v1.NewGreetServiceClient(conn)
	res, err := c.Greet(context.Background(), req)
	return &Greeting{
		Content: res.Result,
	}, err
}
