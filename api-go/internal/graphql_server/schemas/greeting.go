package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/utils"
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
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		firstName := p.Args["firstName"].(string)
		lastName := p.Args["lastName"].(string)
		return utils.GetGreeting(firstName, lastName)
	},
}
