package schemas

import (
	"github.com/graphql-go/graphql"
)

var query = graphql.NewObject(graphql.ObjectConfig{
	Name: "Query",
	Fields: graphql.Fields{
		"seed":                    &seedGraphQLField,
		"debouncedSeed":           &debouncedSeedGraphQLField,
		"currentTime":             &currentTimeGraphQLField,
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
		"setSeed":  &setSeedGraphQLField,
		"signIn":   &signInGraphQLField,
		"adoptDog": &adoptDogGraphQLField,
	},
})

var Schema, _ = graphql.NewSchema(graphql.SchemaConfig{
	Query:    query,
	Mutation: mutation,
})
