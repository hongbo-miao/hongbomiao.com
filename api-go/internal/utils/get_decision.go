package utils

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/policies"
	"github.com/open-policy-agent/opa/rego"
	"github.com/rs/zerolog/log"
)

var policyPath = "policy/rbac.authz.rego"
var defaultQuery = "x = data.rbac.authz.allow"

type input struct {
	User   string `json:"user"`
	Action string `json:"action"`
	Object string `json:"object"`
}

type OPA struct {
	User     string `json:"user"`
	Action   string `json:"action"`
	Object   string `json:"object"`
	Decision bool   `json:"decision"`
}

func getResult(ctx context.Context, query rego.PreparedEvalQuery, input map[string]interface{}) bool {
	results, err := query.Eval(ctx, rego.EvalInput(input))
	if err != nil {
		log.Error().Err(err).Msg("getResult")
	}
	return results[0].Bindings["x"].(bool)
}

func GetDecision(user string, action string, object string) (opa OPA, err error) {
	s := input{
		User:   user,
		Action: action,
		Object: object,
	}

	input := map[string]interface{}{
		"user":   s.User,
		"action": s.Action,
		"object": s.Object,
	}

	p, err := policies.ReadPolicy(policyPath)
	if err != nil {
		log.Error().Err(err).Msg("ReadPolicy")
	}

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
