package utils

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/policies"
	"github.com/open-policy-agent/opa/rego"
	"log"
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

func getResult(ctx context.Context, query rego.PreparedEvalQuery, input map[string]interface{}) (decision bool, err error) {
	results, err := query.Eval(ctx, rego.EvalInput(input))
	if err != nil {
		log.Fatalf("evaluation error: %v", err)
	} else if len(results) == 0 {
		log.Fatal("undefined result", err)
	} else if result, ok := results[0].Bindings["x"].(bool); !ok {
		log.Fatalf("unexpected result type: %v", result)
	}
	return results[0].Bindings["x"].(bool), nil
}

func GetDecision(user string, action string, object string) (opa OPA, err error) {
	s := input{
		User:   user,
		Action: action,
		Object: object,
	}

	input := map[string]interface{}{
		"user":   []string{s.User},
		"action": s.Action,
		"object": s.Object,
	}

	p, err := policies.ReadPolicy(policyPath)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.TODO()
	query, err := rego.New(
		rego.Query(defaultQuery),
		rego.Module(policyPath, string(p)),
	).PrepareForEval(ctx)

	if err != nil {
		log.Fatalf("initial rego error: %v", err)
	}

	decision, _ := getResult(ctx, query, input)
	return OPA{
		User:     user,
		Action:   action,
		Object:   object,
		Decision: decision,
	}, nil
}
