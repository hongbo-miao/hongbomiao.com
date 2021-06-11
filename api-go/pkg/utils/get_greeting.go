package utils

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/api/proto/greet/v1"
	"google.golang.org/grpc"
	"log"
)

type Greeting struct {
	Content string `json:"content"`
}

func GetGreeting(firstName string, lastName string) (greeting Greeting, err error) {
	req := &v1.GreetRequest{
		Greeting: &v1.Greeting{
			FirstName: firstName,
			LastName:  lastName,
		},
	}

	cc, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("could not connect: %v", err)
	}
	defer cc.Close()

	c := v1.NewGreetServiceClient(cc)
	res, err := c.Greet(context.Background(), req)
	return Greeting{
		Content: res.Result,
	}, err
}
