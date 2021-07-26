package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/graphql-go/graphql"
)

var signInGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "SignIn",
	Fields: graphql.Fields{
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
	Resolve: func(p graphql.ResolveParams) (res interface{}, err error) {
		email := p.Args["email"].(string)
		password := p.Args["password"].(string)
		return utils.GetJWTToken(email, password)
	},
}
