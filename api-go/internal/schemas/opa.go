package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/graphql-go/graphql"
	"github.com/rs/zerolog/log"
)

var opaGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "OPA",
	Fields: graphql.Fields{
		"decision": &graphql.Field{
			Type: graphql.Boolean,
		},
	},
})

var opaGraphQLField = graphql.Field{
	Type: opaGraphQLType,
	Args: graphql.FieldConfigArgument{
		"action": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
		"resourceType": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
	},
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		err := utils.CheckGraphQLContextMyID(p)
		if err != nil {
			log.Error().Err(err).Msg("CheckGraphQLContextMyID")
			return nil, err
		}

		myID := p.Context.Value(types.ContextMyIDKey("myID")).(string)
		action := p.Args["action"].(string)
		resourceType := p.Args["resourceType"].(string)
		return utils.GetOPADecision(myID, action, resourceType)
	},
}
