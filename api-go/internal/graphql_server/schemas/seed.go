package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/utils"
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
