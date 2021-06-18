package schemas

import (
	"github.com/graphql-go/graphql"
)

var Query = graphql.NewObject(graphql.ObjectConfig{
	Name: "Query",
	Fields: graphql.Fields{
		"me":       &meGraphQLField,
		"user":     &userGraphQLField,
		"opa":      &opaGraphQLField,
		"greeting": &greetingGraphQLField,
	},
})

var Schema, _ = graphql.NewSchema(graphql.SchemaConfig{
	Query:    Query,
	Mutation: nil,
})
