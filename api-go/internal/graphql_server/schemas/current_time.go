package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/utils"
)

var currentTimeGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "CurrentTime",
	Fields: graphql.Fields{
		"now": &graphql.Field{
			Type: graphql.String,
		},
	},
})

var currentTimeGraphQLField = graphql.Field{
	Type: currentTimeGraphQLType,
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		return utils.GetCurrentTime()
	},
}
