package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/graphql-go/graphql"
)

var greetingGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "Greeting",
	Fields: graphql.Fields{
		"content": &graphql.Field{
			Type: graphql.String,
		},
	},
})

var greetingGraphQLField = graphql.Field{
	Type: greetingGraphQLType,
	Args: graphql.FieldConfigArgument{
		"firstName": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
		"lastName": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
	},
	Resolve: func(p graphql.ResolveParams) (res interface{}, err error) {
		firstName := p.Args["firstName"].(string)
		lastName := p.Args["lastName"].(string)
		return utils.GetGreeting(firstName, lastName)
	},
}
