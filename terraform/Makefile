aws-install: # https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-mac.html
	curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
	sudo installer -pkg AWSCLIV2.pkg -target /
aws-configure: # https://console.aws.amazon.com/iam/home#/security_credentials
	aws configure

terraform-init:
	terraform init
terraform-plan:
	terraform plan
terraform-apply:
	terraform apply
terraform-destroy:
	terraform destroy

terraform-get:
	terraform get
terraform-graph: # http://www.webgraphviz.com
	terraform graph
terraform-login:
	terraform login
terraform-fmt:
	terraform fmt
terraform-validate:
	terraform validate
terraform-show:
	terraform show
terraform-state-list:
	terraform state list
terraform-output:
	terraform output
