package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/graphql-go/graphql"
)

var opaGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "OPA",
	Fields: graphql.Fields{
		"decision": &graphql.Field{
			Type: graphql.String,
		},
	},
})

var opaGraphQLField = graphql.Field{
	Type: opaGraphQLType,
	Args: graphql.FieldConfigArgument{
		"user": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
		"action": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
		"object": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
	},
	Resolve: func(p graphql.ResolveParams) (res interface{}, err error) {
		user := p.Args["user"].(string)
		action := p.Args["action"].(string)
		object := p.Args["object"].(string)
		return utils.GetDecision(user, action, object)
	},
}
