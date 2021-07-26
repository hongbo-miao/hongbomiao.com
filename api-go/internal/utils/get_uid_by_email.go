package utils

import (
	"context"
	"github.com/buger/jsonparser"
	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
)

func GetUIDByEmail(email string) string {
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

	q := `query user($email: string) {
	  user(func: eq(email, $email)) {
		uid
	  }
    }`
	req := &api.Request{
		Query: q,
		Vars:  map[string]string{"$email": email},
	}
	res, err := txn.Do(ctx, req)
	if err != nil {
		log.Error().Err(err).Msg("txn.Do")
	}

	uid, err := jsonparser.GetString(res.Json, "user", "[0]", "uid")
	if err != nil {
		log.Error().
			Err(err).
			Bytes("res.Json", res.Json).
			Msg("jsonparser.GetString")
	}

	return uid
}
