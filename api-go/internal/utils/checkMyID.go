package utils

import (
	"errors"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/types"
	"github.com/graphql-go/graphql"
)

func CheckGraphQLContextMyID(p graphql.ResolveParams) error {
	if p.Context.Value(types.ContextMyIDKey("myID")).(string) != "" {
		return nil
	}
	return errors.New("not authorized")
}
