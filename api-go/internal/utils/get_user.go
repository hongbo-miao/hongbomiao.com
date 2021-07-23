package utils

import (
	"context"
	"github.com/buger/jsonparser"
	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
)

type User struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

func GetUser(id string) (user User, err error) {
	var config = GetConfig()
	conn, err := grpc.Dial(config.DgraphHost+":"+config.DgraphGRPCPort, grpc.WithInsecure())
	if err != nil {
		log.Error().Err(err).Msg("grpc.Dial")
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

	q := `query user($uid: string) {
	  user(func: uid($uid)) {
		name
	  }
    }`
	req := &api.Request{
		Query: q,
		Vars:  map[string]string{"$uid": id},
	}
	res, err := txn.Do(ctx, req)
	if err != nil {
		log.Error().Err(err).Msg("txn.Do")
	}

	name, err := jsonparser.GetString(res.Json, "user", "[0]", "name")
	if err != nil {
		log.Error().
			Err(err).
			Bytes("res.Json", res.Json).
			Msg("jsonparser.GetString")
	}

	return User{ID: id, Name: name}, nil
}
