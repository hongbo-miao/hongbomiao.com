docker-build-development:
	cd .. && docker build --file=api-node/Dockerfile.development --tag=ghcr.io/hongbo-miao/hm-api-node-dev:latest .
docker-push-development:
	docker push ghcr.io/hongbo-miao/hm-api-node-dev:latest
docker-run-development:
	docker run --publish=5000:5000 --name=hm-api-python --rm --env-file=./api-node/.env.development.local.example.docker ghcr.io/hongbo-miao/hm-api-node-dev:latest

docker-build-production:
	cd .. && docker build --file=api-node/Dockerfile --tag=ghcr.io/hongbo-miao/hm-api-node:latest .
docker-push-production:
	docker push ghcr.io/hongbo-miao/hm-api-node:latest
docker-run-production:
	docker run --publish=5000:5000 --name=hm-api-python --rm --env-file=./api-node/.env.production.local.example ghcr.io/hongbo-miao/hm-api-node:latest
