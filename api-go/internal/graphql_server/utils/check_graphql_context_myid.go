package utils

import (
	"errors"
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/types"
)

func CheckGraphQLContextMyID(p graphql.ResolveParams) error {
	if p.Context.Value(types.ContextKey("myID")).(string) == "" {
		return errors.New("no myID")
	}
	return nil
}
