package schemas

import (
	"github.com/graphql-go/graphql"
)

var query = graphql.NewObject(graphql.ObjectConfig{
	Name: "Query",
	Fields: graphql.Fields{
		"opa":  &opaGraphQLField,
		"opal": &opalGraphQLField,
	},
})

var Schema, _ = graphql.NewSchema(graphql.SchemaConfig{
	Query: query,
})
