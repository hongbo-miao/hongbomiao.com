package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/utils"
	"github.com/graphql-go/graphql"
	"github.com/rs/zerolog/log"
)

var meGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "Me",
	Fields: graphql.Fields{
		"id": &graphql.Field{
			Type: graphql.ID,
		},
		"name": &graphql.Field{
			Type: graphql.String,
		},
		"age": &graphql.Field{
			Type: graphql.Int,
		},
		"email": &graphql.Field{
			Type: graphql.String,
		},
	},
})

var meGraphQLField = graphql.Field{
	Type: meGraphQLType,
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		err := utils.CheckGraphQLContextMyID(p)
		if err != nil {
			log.Error().Err(err).Msg("CheckGraphQLContextMyID")
			return nil, err
		}

		myID := p.Context.Value(types.ContextMyIDKey("myID")).(string)
		return utils.GetMe(myID)
	},
}
