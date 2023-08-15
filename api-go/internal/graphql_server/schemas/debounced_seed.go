package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/utils"
)

var debouncedSeedGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "DebouncedSeed",
	Fields: graphql.Fields{
		"seedNumber": &graphql.Field{
			Type: graphql.Int,
		},
	},
})

var debouncedSeedGraphQLField = graphql.Field{
	Type: debouncedSeedGraphQLType,
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		return utils.GetDebouncedSeed()
	},
}
