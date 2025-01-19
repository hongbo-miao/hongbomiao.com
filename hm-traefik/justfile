traefik:
	traefik --configfile=traefik.yaml

docker-build:
	cd .. && \
	docker build --file=traefik/Dockerfile --tag=ghcr.io/hongbo-miao/hm-traefik:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-traefik:latest
