package utils

import (
	"errors"
	"github.com/graphql-go/graphql"
)

func CheckGraphQLContextMyID(p graphql.ResolveParams) error {
	if p.Context.Value("myID").(string) != "" {
		return nil
	}
	return errors.New("not authorized")
}
