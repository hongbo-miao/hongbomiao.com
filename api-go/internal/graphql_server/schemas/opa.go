package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/utils"
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
		"resource": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.String),
		},
	},
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		err := utils.CheckGraphQLContextMyID(p)
		if err != nil {
			log.Error().Err(err).Msg("CheckGraphQLContextMyID")
			return nil, err
		}

		myID := p.Context.Value(types.ContextKey("myID")).(string)
		action := p.Args["action"].(string)
		resource := p.Args["resource"].(string)
		return utils.GetOPADecision(myID, action, resource)
	},
}
