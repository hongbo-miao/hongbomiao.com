package schemas

import (
	"github.com/graphql-go/graphql"
)

var query = graphql.NewObject(graphql.ObjectConfig{
	Name: "Query",
	Fields: graphql.Fields{
		"opa":                     &opaGraphQLField,
		"opal":                    &opalGraphQLField,
		"me":                      &meGraphQLField,
		"user":                    &userGraphQLField,
		"greeting":                &greetingGraphQLField,
		"dog":                     &dogGraphQLField,
		"twitterHashtag":          &twitterHashtagGraphQLField,
		"trendingTwitterHashtags": &trendingTwitterHashtagsGraphQLField,
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
