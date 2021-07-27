package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/types"
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
		"action": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
		"resourceType": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
	},
	Resolve: func(p graphql.ResolveParams) (res interface{}, err error) {
		err = utils.CheckGraphQLContextMyID(p)
		if err != nil {
			return nil, err
		}
		myID := p.Context.Value(types.ContextMyIDKey("myID")).(string)
		action := p.Args["action"].(string)
		resourceType := p.Args["resourceType"].(string)
		return utils.GetOPALDecision(myID, action, resourceType)
	},
}
