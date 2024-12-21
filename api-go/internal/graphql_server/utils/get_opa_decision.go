package utils

import (
	"context"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/policies"
	"github.com/open-policy-agent/opa/v1/rego"
	"github.com/open-policy-agent/opa/v1/storage/inmem"
	"github.com/open-policy-agent/opa/v1/util"
	"github.com/rs/zerolog/log"
)

var policyPath = "app.rbac.rego"
var defaultQuery = "x = data.app.rbac.allow"

type OPA struct {
	Decision bool `json:"decision"`
}

func getResult(ctx context.Context, query rego.PreparedEvalQuery, input map[string]interface{}) bool {
	results, err := query.Eval(ctx, rego.EvalInput(input))
	if err != nil {
		log.Error().Err(err).Msg("getResult")
	}
	return results[0].Bindings["x"].(bool)
}

func GetOPADecision(uid string, action string, resource string) (*OPA, error) {
	me, err := GetMe(uid)
	if err != nil {
		log.Error().Err(err).Msg("GetMe")
		return nil, err
	}

	input := map[string]interface{}{
		"roles":    me.Roles,
		"action":   action,
		"resource": resource,
	}

	data := policies.ReadData()
	var json map[string]interface{}
	err = util.UnmarshalJSON(data, &json)
	if err != nil {
		log.Error().Err(err).Msg("UnmarshalJSON")
		return nil, err
	}
	store := inmem.NewFromObject(json)
	policy := policies.ReadPolicy()
	ctx := context.TODO()

	query, err := rego.New(
		rego.Query(defaultQuery),
		rego.Store(store),
		rego.Module(policyPath, string(policy)),
	).PrepareForEval(ctx)

	if err != nil {
		log.Error().Err(err).Msg("PrepareForEval")
		return nil, err
	}

	decision := getResult(ctx, query, input)
	return &OPA{
		Decision: decision,
	}, nil
}
