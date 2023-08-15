package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/types"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	"github.com/redis/go-redis/v9"
)

var trendingTwitterHashtagsGraphQLField = graphql.Field{
	Type: graphql.NewList(twitterHashtagGraphQLType),
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		rdb := p.Context.Value(types.ContextKey("rdb")).(*redis.Client)
		return utils.GetTrendingTwitterHashtags(rdb)
	},
}
