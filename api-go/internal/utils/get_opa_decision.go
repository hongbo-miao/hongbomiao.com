package utils

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/policies"
	"github.com/open-policy-agent/opa/rego"
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

func GetOPADecision(user string, action string, resourceType string, object string) (opa OPA, err error) {
	input := map[string]interface{}{
		"user":   user,
		"action": action,
		"type":   resourceType,
		"object": object,
	}

	p := policies.ReadPolicy()

	ctx := context.TODO()
	query, err := rego.New(
		rego.Query(defaultQuery),
		rego.Module(policyPath, string(p)),
	).PrepareForEval(ctx)

	if err != nil {
		log.Error().Err(err).Msg("PrepareForEval")
	}

	decision := getResult(ctx, query, input)
	return OPA{
		Decision: decision,
	}, nil
}
