package utils

import (
	"context"
	"github.com/buger/jsonparser"
	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
)

type Me struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Age      int    `json:"age"`
	Email    string `json:"email"`
	JWTToken string `json:"jwtToken"`
}

func GetMe(id string) (*Me, error) {
	var config = GetConfig()
	conn, err := grpc.Dial(config.DgraphHost+":"+config.DgraphGRPCPort, grpc.WithInsecure())
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

	dgraphClient := dgo.NewDgraphClient(api.NewDgraphClient(conn))
	ctx := context.Background()
	txn := dgraphClient.NewTxn()
	defer func(txn *dgo.Txn, ctx context.Context) {
		err := txn.Discard(ctx)
		if err != nil {
			log.Error().Err(err).Msg("txn.Discard")
		}
	}(txn, ctx)

	q := `query Me($uid: string) {
	  me(func: uid($uid)) {
		name
		age
		email
	  }
    }`
	req := &api.Request{
		Query: q,
		Vars:  map[string]string{"$uid": id},
	}
	res, err := txn.Do(ctx, req)
	if err != nil {
		log.Error().Err(err).Msg("txn.Do")
		return nil, err
	}

	name, err := jsonparser.GetString(res.Json, "me", "[0]", "name")
	if err != nil {
		log.Error().Err(err).Bytes("res.Json", res.Json).Msg("jsonparser.GetString")
		return nil, err
	}
	age, err := jsonparser.GetInt(res.Json, "me", "[0]", "age")
	if err != nil {
		log.Error().Err(err).Bytes("res.Json", res.Json).Msg("jsonparser.GetString")
		return nil, err
	}
	email, err := jsonparser.GetString(res.Json, "me", "[0]", "email")
	if err != nil {
		log.Error().Err(err).Bytes("res.Json", res.Json).Msg("jsonparser.GetString")
		return nil, err
	}

	return &Me{
		ID:    id,
		Name:  name,
		Age:   int(age),
		Email: email,
	}, nil
}
