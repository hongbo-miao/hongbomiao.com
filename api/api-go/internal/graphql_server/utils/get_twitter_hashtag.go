package utils

import (
	"context"
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/types"
	"github.com/rs/zerolog/log"
	"github.com/valkey-io/valkey-go"
	"strconv"
)

func GetTwitterHashtag(text string, valkeyClient valkey.Client) (*types.TwitterHashtag, error) {
	ctx := context.Background()

	result, err := valkeyClient.Do(ctx, valkeyClient.B().Hget().Key("trending-twitter-hashtags").Field(text).Build()).ToString()
	if err != nil {
		if valkey.IsValkeyNil(err) {
			return &types.TwitterHashtag{
				Text:  text,
				Count: 0,
			}, nil
		}
		log.Error().Err(err).Msg("valkeyClient.Do Hget")
		return nil, err
	}

	count, err := strconv.Atoi(result)
	if err != nil {
		return nil, err
	}

	return &types.TwitterHashtag{
		Text:  text,
		Count: count,
	}, nil
}
