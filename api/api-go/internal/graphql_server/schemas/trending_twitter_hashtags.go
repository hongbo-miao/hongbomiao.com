package schemas

import (
	"github.com/graphql-go/graphql"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/types"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/utils"
	"github.com/valkey-io/valkey-go"
)

var trendingTwitterHashtagsGraphQLField = graphql.Field{
	Type: graphql.NewList(twitterHashtagGraphQLType),
	Resolve: func(p graphql.ResolveParams) (interface{}, error) {
		valkeyClient := p.Context.Value(types.ContextKey("valkeyClient")).(valkey.Client)
		return utils.GetTrendingTwitterHashtags(valkeyClient)
	},
}
