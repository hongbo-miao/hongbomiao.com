package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/types"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/utils"
	"github.com/redis/go-redis/v9"
)

var twitterHashtagGraphQLType = graphql.NewObject(graphql.ObjectConfig{
	Name: "TwitterHashtag",
	Fields: graphql.Fields{
		"text": &graphql.Field{
			Type: graphql.ID,
		},
		"count": &graphql.Field{
			Type: graphql.Int,
		},
	},
})

var twitterHashtagGraphQLField = graphql.Field{
	Type: twitterHashtagGraphQLType,
	Args: graphql.FieldConfigArgument{
		"text": &graphql.ArgumentConfig{
			Type: graphql.NewNonNull(graphql.ID),
		},
	},
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		rdb := p.Context.Value(types.ContextKey("rdb")).(*redis.Client)
		text := p.Args["text"].(string)
		return utils.GetTwitterHashtag(text, rdb)
	},
}
