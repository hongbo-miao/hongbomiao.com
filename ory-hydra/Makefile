CLIENT_SECRET="S3xGn9eZseWNJYwr"

hydra-create-client:
	hydra clients create \
		--endpoint=http://localhost:4445/ \
		--id=hm-hasura-client \
		--secret=${CLIENT_SECRET} \
		--grant-types=client_credentials

hydra-generate-token:
	hydra token client \
		--endpoint=http://localhost:4444/ \
		--client-id=hm-hasura-client \
		--client-secret=${CLIENT_SECRET}

hydra-introspect-token:
	hydra token introspect \
		--endpoint http://localhost:4445/ \
		JEVbK47mqINg1p_ZIJSsVc_umqe5Byfi2bMMyG3NXtQ.Ac9QUsXZTt97b6OAp7MioDZx3zI4iy1fM4WzixFHbMc
