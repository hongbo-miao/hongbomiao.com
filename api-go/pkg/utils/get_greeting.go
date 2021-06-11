package utils

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/greet/greetpb"
	"google.golang.org/grpc"
	"log"
)

type Greeting struct {
	Content string `json:"content"`
}

func GetGreeting(firstName string, lastName string) (greeting Greeting, err error) {
	req := &greetpb.GreetRequest{
		Greeting: &greetpb.Greeting{
			FirstName: firstName,
			LastName:  lastName,
		},
	}

	cc, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("could not connect: %v", err)
	}
	defer cc.Close()

	c := greetpb.NewGreetServiceClient(cc)
	res, err := c.Greet(context.Background(), req)
	return Greeting{
		Content: res.Result,
	}, err
}
