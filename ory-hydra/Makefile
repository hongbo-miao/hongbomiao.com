CLIENT_SECRET="NKvHvuzcK3T4TNU5q47VzRnGzhTzqdvQkJgU4bKSTxSCBz8QhZZ753sKUAskAHtz"

hydra-admin-create-client:
	hydra clients create \
		--endpoint=http://localhost:4445/ \
		--id=hm-hasura-user \
		--secret=${CLIENT_SECRET} \
		--grant-types=client_credentials
hydra-admin-delete-client:
	hydra clients delete hm-hasura-user \
		--endpoint=http://localhost:4445/

hydra-public-generate-token:
	hydra token client \
		--endpoint=http://localhost:4444/ \
		--client-id=hm-hasura-user \
		--client-secret=${CLIENT_SECRET}

hydra-admin-introspect-token:
	hydra token introspect \
		--endpoint=http://localhost:4445/ \
		xxx
