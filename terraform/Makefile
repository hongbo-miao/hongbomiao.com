# https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-mac.html
aws-install:
	curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
	sudo installer -pkg AWSCLIV2.pkg -target /
# https://console.aws.amazon.com/iam/home#/security_credentials
aws-configure:
	aws configure --profile=hm
	# AWS Access Key ID: xxx
	# AWS Secret Access Key: xxx
	# Default region name: us-west-2
	# Default output format:

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
terraform-validate:
	terraform validate
terraform-show:
	terraform show
terraform-state-list:
	terraform state list
terraform-output:
	terraform output
