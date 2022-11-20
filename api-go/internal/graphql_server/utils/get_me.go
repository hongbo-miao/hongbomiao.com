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

type Me struct {
	ID       string   `json:"id"`
	Name     string   `json:"name"`
	Age      int      `json:"age"`
	Email    string   `json:"email"`
	Roles    []string `json:"roles"`
	JWTToken string   `json:"jwtToken"`
}

func GetMe(id string) (*Me, error) {
	config := GetConfig()
	conn, err := grpc.Dial(
		config.DgraphHost+":"+config.DgraphGRPCPort,
		grpc.WithUnaryInterceptor(apmgrpc.NewUnaryClientInterceptor()),
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
		roles
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
		log.Error().Err(err).Bytes("res.Json", res.Json).Msg("jsonparser.GetString name")
		return nil, err
	}

	age, err := jsonparser.GetInt(res.Json, "me", "[0]", "age")
	if err != nil {
		log.Error().Err(err).Bytes("res.Json", res.Json).Msg("jsonparser.GetInt age")
		return nil, err
	}

	email, err := jsonparser.GetString(res.Json, "me", "[0]", "email")
	if err != nil {
		log.Error().Err(err).Bytes("res.Json", res.Json).Msg("jsonparser.GetString email")
		return nil, err
	}

	var roles []string
	_, err = jsonparser.ArrayEach(res.Json, func(value []byte, dataType jsonparser.ValueType, offset int, err error) {
		roles = append(roles, string(value))
	}, "me", "[0]", "roles")
	if err != nil {
		log.Error().Err(err).Bytes("res.Json", res.Json).Msg("jsonparser.ArrayEach")
		return nil, err
	}

	return &Me{
		ID:    id,
		Name:  name,
		Age:   int(age),
		Email: email,
		Roles: roles,
	}, nil
}
