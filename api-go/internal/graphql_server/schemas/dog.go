package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/types"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	"github.com/rs/zerolog/log"
)

var dogGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "Dog",
	Fields: graphql.Fields{
		"id": &graphql.Field{
			Type: graphql.ID,
		},
		"name": &graphql.Field{
			Type: graphql.String,
		},
	},
})

var dogGraphQLField = graphql.Field{
	Type: dogGraphQLType,
	Args: graphql.FieldConfigArgument{
		"id": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.ID),
		},
	},
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		err := utils.CheckGraphQLContextMyID(p)
		if err != nil {
			log.Error().Err(err).Msg("CheckGraphQLContextMyID")
			return nil, err
		}

		myID := p.Context.Value(types.ContextKey("myID")).(string)
		dogID := p.Args["id"].(string)
		return utils.GetDog(myID, dogID)
	},
}
