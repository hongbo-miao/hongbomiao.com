CLIENT_SECRET="S3xGn9eZseWNJYwr"

hydra-admin-create-client:
	hydra clients create \
		--endpoint=http://localhost:4445/ \
		--id=hm-hasura-client \
		--secret=${CLIENT_SECRET} \
		--grant-types=client_credentials

hydra-public-generate-token:
	hydra token client \
		--endpoint=http://localhost:4444/ \
		--client-id=hm-hasura-client \
		--client-secret=${CLIENT_SECRET}

hydra-admin-introspect-token:
	hydra token introspect \
		--endpoint=http://localhost:4445/ \
		xxx
