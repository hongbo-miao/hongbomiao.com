package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/utils"
	"github.com/graphql-go/graphql"
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

		myID := p.Context.Value(types.ContextMyIDKey("myID")).(string)
		dogID := p.Args["id"].(string)
		return utils.GetDog(myID, dogID)
	},
}
