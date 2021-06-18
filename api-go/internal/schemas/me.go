package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/graphql-go/graphql"
)

var meGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "Me",
	Fields: graphql.Fields{
		"id": &graphql.Field{
			Type: graphql.ID,
		},
		"firstName": &graphql.Field{
			Type: graphql.String,
		},
		"lastName": &graphql.Field{
			Type: graphql.String,
		},
		"name": &graphql.Field{
			Type: graphql.String,
		},
		"bio": &graphql.Field{
			Type: graphql.String,
		},
	},
})

var meGraphQLField = graphql.Field{
	Type: meGraphQLType,
	Resolve: func(p graphql.ResolveParams) (res interface{}, err error) {
		return utils.GetMe()
	},
}
