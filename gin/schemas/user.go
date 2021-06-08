package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/gin/utils"
	"github.com/graphql-go/graphql"
)

var userGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "User",
	Fields: graphql.Fields{
		"id": &graphql.Field{
			Type: graphql.ID,
		},
		"name": &graphql.Field{
			Type: graphql.String,
		},
	},
})

var userGraphQLField = graphql.Field{
	Type: userGraphQLType,
	Args: graphql.FieldConfigArgument{
		"id": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.ID),
		},
	},
	Resolve: func(p graphql.ResolveParams) (res interface{}, err error) {
		id := p.Args["id"].(string)
		return utils.GetUser(id)
	},
}
