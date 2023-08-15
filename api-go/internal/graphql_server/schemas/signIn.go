package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/utils"
)

var signInGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "SignIn",
	Fields: graphql.Fields{
		"id": &graphql.Field{
			Type: graphql.ID,
		},
		"name": &graphql.Field{
			Type: graphql.String,
		},
		"jwtToken": &graphql.Field{
			Type: graphql.String,
		},
	},
})

var signInGraphQLField = graphql.Field{
	Type: signInGraphQLType,
	Args: graphql.FieldConfigArgument{
		"email": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
		"password": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
	},
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		email := p.Args["email"].(string)
		password := p.Args["password"].(string)
		return utils.SignIn(email, password)
	},
}
