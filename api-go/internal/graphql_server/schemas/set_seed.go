package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	"github.com/graphql-go/graphql"
)

var setSeedGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "SetSeed",
	Fields: graphql.Fields{
		"seedNumber": &graphql.Field{
			Type: graphql.Int,
		},
	},
})

var setSeedGraphQLField = graphql.Field{
	Type: setSeedGraphQLType,
	Args: graphql.FieldConfigArgument{
		"n": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.Int),
		},
	},
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		n := p.Args["n"].(int)
		return utils.SetSeed(n)
	},
}
