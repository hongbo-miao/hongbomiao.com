package schemas

import (
	"github.com/graphql-go/graphql"
)

var query = graphql.NewObject(graphql.ObjectConfig{
	Name: "Query",
	Fields: graphql.Fields{
		"me":       &meGraphQLField,
		"user":     &userGraphQLField,
		"greeting": &greetingGraphQLField,
		"dog":      &dogGraphQLField,
	},
})

var mutation = graphql.NewObject(graphql.ObjectConfig{
	Name: "Mutation",
	Fields: graphql.Fields{
		"signIn":   &signInGraphQLField,
		"adoptDog": &adoptDogGraphQLField,
	},
})

var Schema, _ = graphql.NewSchema(graphql.SchemaConfig{
	Query:    query,
	Mutation: mutation,
})
