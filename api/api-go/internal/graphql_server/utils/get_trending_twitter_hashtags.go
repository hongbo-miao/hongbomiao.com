package utils

import (
	"context"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/types"
	"github.com/rs/zerolog/log"
	"github.com/valkey-io/valkey-go"
	"strconv"
)

func GetTrendingTwitterHashtags(valkeyClient valkey.Client) ([]types.TwitterHashtag, error) {
	ctx := context.Background()
	var trendingTwitterHashtags []types.TwitterHashtag

	result, err := valkeyClient.Do(ctx, valkeyClient.B().Hgetall().Key("trending-twitter-hashtags").Build()).AsStrMap()
	if err != nil {
		if valkey.IsValkeyNil(err) {
			return trendingTwitterHashtags, nil
		}
		log.Error().Err(err).Msg("valkeyClient.Do Hgetall")
		return nil, err
	}

	for k, v := range result {
		count, err := strconv.Atoi(v)
		if err != nil {
			return nil, err
		}
		trendingTwitterHashtags = append(trendingTwitterHashtags, types.TwitterHashtag{
			Text:  k,
			Count: count,
		})
	}

	return trendingTwitterHashtags, nil
}
