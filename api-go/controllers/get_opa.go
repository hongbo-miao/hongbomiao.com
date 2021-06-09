package controllers

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/policies"
	"github.com/gin-gonic/gin"
	"github.com/open-policy-agent/opa/rego"
	"log"
	"net/http"
)

var policyPath = "policy/rbac.authz.rego"
var defaultQuery = "x = data.rbac.authz.allow"

type input struct {
	User   string `json:"user"`
	Action string `json:"action"`
	Object string `json:"object"`
}

func getResult(ctx context.Context, query rego.PreparedEvalQuery, input map[string]interface{}) (bool, error) {
	results, err := query.Eval(ctx, rego.EvalInput(input))
	if err != nil {
		log.Fatalf("evaluation error: %v", err)
	} else if len(results) == 0 {
		log.Fatal("undefined result", err)
		// Handle undefined result.
	} else if result, ok := results[0].Bindings["x"].(bool); !ok {
		log.Fatalf("unexpected result type: %v", result)
	}

	return results[0].Bindings["x"].(bool), nil
}

func GetOPA(c *gin.Context) {
	s := input{
		User:   "jack",
		Action: "read",
		Object: "server123",
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

	ok, _ := getResult(ctx, query, input)
	log.Println(ok)
	c.JSON(http.StatusOK, gin.H{
		"data": ok,
	})
}
