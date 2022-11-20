package utils

import (
	"context"
	"github.com/buger/jsonparser"
	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"
	"github.com/rs/zerolog/log"
	"go.elastic.co/apm/module/apmgrpc/v2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func VerifyPassword(uid string, password string) (bool, error) {
	config := GetConfig()
	conn, err := grpc.Dial(
		config.DgraphHost+":"+config.DgraphGRPCPort,
		grpc.WithUnaryInterceptor(apmgrpc.NewUnaryClientInterceptor()),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Error().Err(err).Msg("grpc.Dial")
		return false, err
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

	q := `query VerifyPassword($uid: string, $password: string) {
	  verifyPassword(func: uid($uid)) {
        checkpwd(password, $password)
	  }
    }`
	req := &api.Request{
		Query: q,
		Vars: map[string]string{
			"$uid":      uid,
			"$password": password,
		},
	}
	res, err := txn.Do(ctx, req)
	if err != nil {
		log.Error().Err(err).Msg("txn.Do")
		return false, err
	}

	isPasswordValid, err := jsonparser.GetBoolean(res.Json, "verifyPassword", "[0]", "checkpwd(password)")
	if err != nil {
		log.Error().
			Err(err).
			Bytes("res.Json", res.Json).
			Msg("jsonparser.GetString")
		return false, err
	}

	return isPasswordValid, nil
}
