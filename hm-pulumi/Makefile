poetry-env-use:
	poetry env use python3.11
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

pulumi-login:
	pulumi login s3://hongbomiao-pulumi-bucket
pulumi-init:
	pulumi new aws-python

pulumi-up:
	export PULUMI_CONFIG_PASSPHRASE_FILE="passphrase.txt" && pulumi up
pulumi-down:
	export PULUMI_CONFIG_PASSPHRASE_FILE="passphrase.txt" && pulumi destroy

pulumi-stack-list:
	pulumi stack ls
pulumi-stack-init:
	pulumi stack init development

pulumi-convert:
	pulumi convert --from=terraform --language=python --out=my-pulumi --generate-only
