package schemas

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/utils"
	"github.com/go-redis/redis/v8"
	"github.com/graphql-go/graphql"
)

var trendingTwitterHashtagsGraphQLField = graphql.Field{
	Type: graphql.NewList(twitterHashtagGraphQLType),
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		rdb := p.Context.Value(types.ContextKey("rdb")).(*redis.Client)
		return utils.GetTrendingTwitterHashtags(rdb)
	},
}
