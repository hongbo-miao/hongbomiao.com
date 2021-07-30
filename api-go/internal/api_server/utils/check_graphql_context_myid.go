package utils

import (
	"errors"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/types"
	"github.com/graphql-go/graphql"
)

func CheckGraphQLContextMyID(p graphql.ResolveParams) error {
	if p.Context.Value(types.ContextMyIDKey("myID")).(string) == "" {
		return errors.New("no myID")
	}
	return nil
}
