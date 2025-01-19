# https://hasura.io/docs/latest/graphql/core/hasura-cli/index/

hasura-init:
	hasura init .
hasura-console:
	hasura console

hasura-migrate-create:
	hasura migrate create "create_role_table"
hasura-migrate-create-from-server:
	hasura migrate create "init" --from-server
hasura-migrate-status:
	hasura migrate status
hasura-migrate-apply:
	hasura migrate apply
hasura-migrate-apply-skip-execution:
	hasura migrate apply --skip-execution --version=xxxxxxxxxxxxx
hasura-migrate-apply-goto:
	hasura migrate apply --goto --version=xxxxxxxxxxxxx
hasura-migrate-squash:
	hasura migrate squash --name=xxx --from=xxxxxxxxxxxxx

hasura-metadata-export:
	hasura metadata export
hasura-metadata-diff:
	hasura metadata diff
hasura-metadata-apply:
	hasura metadata apply

hasura-seed-create:
	hasura seed create "xxx_seed"
hasura-seed-create-from-table:
	hasura seed create "role_seed" --from-table=role
hasura-seed-apply:
	hasura seed apply

copy-migrations-metadada-to-k8s-data-folder:
	cd .. && \
	cp -R hasura-graphql-engine/migrations/ kubernetes/data/hasura/hasura-graphql-engine/migrations/ && \
	cp -R hasura-graphql-engine/metadata/ kubernetes/data/hasura/hasura-graphql-engine/metadata/
