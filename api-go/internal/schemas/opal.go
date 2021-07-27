package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/graphql-go/graphql"
)

var opalGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "OPAL",
	Fields: graphql.Fields{
		"decision": &graphql.Field{
			Type: graphql.Boolean,
		},
	},
})

var opalGraphQLField = graphql.Field{
	Type: opalGraphQLType,
	Args: graphql.FieldConfigArgument{
		"user": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
		"action": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
		"resourceType": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
	},
	Resolve: func(p graphql.ResolveParams) (res interface{}, err error) {
		user := p.Args["user"].(string)
		action := p.Args["action"].(string)
		resourceType := p.Args["resourceType"].(string)
		return utils.GetOPALDecision(user, action, resourceType)
	},
}
