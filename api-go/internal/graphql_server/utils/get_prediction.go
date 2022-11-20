package utils

import (
	"context"
	"encoding/json"
	torchserve "github.com/Hongbo-Miao/hongbomiao.com/api-go/api/graphql_server/proto/torchserve/v1"
	"github.com/rs/zerolog/log"
	"go.elastic.co/apm/module/apmgrpc/v2"
	"go.opencensus.io/plugin/ocgrpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"io/ioutil"
	"mime/multipart"
)

type Prediction map[string]float64

func GetPrediction(fileHeader *multipart.FileHeader) (*Prediction, error) {
	file, err := fileHeader.Open()
	if err != nil {
		log.Error().Err(err).Msg("file.Open")
		return nil, err
	}
	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		log.Error().Err(err).Msg("ioutil.ReadAll")
		return nil, err
	}

	input := make(map[string][]byte)
	input["data"] = bytes

	req := &torchserve.PredictionsRequest{
		ModelName:    "densenet161",
		ModelVersion: "1.0",
		Input:        input,
	}

	config := GetConfig()
	conn, err := grpc.Dial(
		config.TorchServeGRPCHost+":"+config.TorchServeGRPCPort,
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

	c := torchserve.NewInferenceAPIsServiceClient(conn)
	res, err := c.Predictions(context.Background(), req)
	if err != nil {
		log.Error().Err(err).Msg("c.Predictions")
		return nil, err
	}

	var prediction Prediction
	err = json.Unmarshal(res.Prediction, &prediction)
	if err != nil {
		log.Error().Err(err).Msg("json.Unmarshal")
	}

	return &prediction, nil
}
