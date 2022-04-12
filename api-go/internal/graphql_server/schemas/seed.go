package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	"github.com/graphql-go/graphql"
)

var seedGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "Seed",
	Fields: graphql.Fields{
		"seedNumber": &graphql.Field{
			Type: graphql.Int,
		},
	},
})

var seedGraphQLField = graphql.Field{
	Type: seedGraphQLType,
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		return utils.GetSeed()
	},
}
