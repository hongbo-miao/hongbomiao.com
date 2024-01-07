# https://docs.temporal.io/docs/tctl/environment-variables
TEMPORAL_CLI_NAMESPACE := default
TEMPORAL_CLI_SHOW_STACKS := 1

export TEMPORAL_CLI_NAMESPACE
export TEMPORAL_CLI_SHOW_STACKS

temporal--namespace-register:
	tctl --namespace=default namespace register
temporal-namespace-describe:
	tctl namespace describe
temporal-workflow-run:
	tctl workflow run --taskqueue=hello-world --workflow_type=Workflow --input='"temporal"'
temporal-workflow-start:
	tctl workflow start --taskqueue=hello-world --workflow_type=Workflow --input='"temporal"'
temporal-taskqueue-describe:
	tctl taskqueue describe --taskqueue=hello-world
