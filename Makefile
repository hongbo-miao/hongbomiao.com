# Kubernetes
kubernetes-setup:
	sh kubernetes/bin/setup.sh
kubernetes-clean:
	sh kubernetes/bin/clean.sh

# Local
local-setup:
	sh bin/setup.sh
local-build:
	sh bin/build.sh
local-clean:
	sh bin/clean.sh

# Docker
docker-login:
	docker login
docker-sh:
	docker run --rm -it hm-graphql-server sh
docker-ps:
	docker ps
docker-ps-all:
	docker ps --all
docker-rmi:
	docker rmi --force IMAGE_ID
docker-prune:
	docker system prune

# Docker Compose
docker-compose-build:
	docker-compose --file=docker-compose.development.yaml build
	docker-compose --file=docker-compose.cypress.yaml build
docker-compose-up:
	docker-compose --file=docker-compose.development.yaml up --detach
	docker-compose --file=docker-compose.cypress.yaml up --detach
docker-compose-stop:
	docker-compose --file=docker-compose.development.yaml stop
	docker-compose --file=docker-compose.cypress.yaml stop
docker-compose-down:
	docker-compose --file=docker-compose.development.yaml down --volumes
	docker-compose --file=docker-compose.cypress.yaml down --volumes

# JupterLab
jupyter-lab:
	jupyter-lab

# Ruby
rbenv-list-latest-stable-versions:
	rbenv install -l
rbenv-install:
	rbenv install 3.1.3

bundle-init:
	bundle init
bundle-install:
	bundle install
bundle-add:
	bundle add xxx
bundle-update:
	bundle update
bundle-lock:
	bundle lock --add-platform x86_64-linux

lint-ruby:
	bundle exec rubocop
lint-ruby-fix:
	bundle exec rubocop --autocorrect-all

# Python
pyenv-list-versions:
	pyenv versions
pyenv-install:
	pyenv install 3.11
pyenv-local:
	pyenv local 3.11

poetry-self-update:
	poetry self update
poetry-env-list:
	poetry env list
poetry-env-remove:
	poetry env remove xxx
poetry-env-use:
	poetry env use 3.11
poetry-lock:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

check-static-type-python:
	poetry run poe check-static-type-python --package=api-python
	poetry run poe check-static-type-python --package=convolutional-neural-network
	poetry run poe check-static-type-python --package=graph-neural-network
	poetry run poe check-static-type-python --package=hm-locust
	poetry run poe check-static-type-python --package=hm-opal-client
	poetry run poe check-static-type-python --package=hm-prefect
	poetry run poe check-static-type-python --package=hm-pyspark
	poetry run poe check-static-type-python --package=quantum-computing
lint-python-black:
	poetry run poe lint-python-black
lint-python-black-fix:
	poetry run poe lint-python-black-fix
lint-python-flake8:
	poetry run poe lint-python-flake8
lint-python-isort:
	poetry run poe lint-python-isort
lint-python-isort-fix:
	poetry run poe lint-python-isort-fix

# Lint
lint-dockerfile:
	hadolint $$(git ls-files "**/Dockerfile*")
lint-shell:
	shellcheck $$(git ls-files "**/*.sh")
lint-kubernetes:
	kubeconform \
		-kubernetes-version=1.23.4 \
		-ignore-filename-pattern=".*trafficsplit.yaml" \
		-ignore-filename-pattern=".*my-values.yaml" \
		-ignore-filename-pattern="kubernetes/manifests/argocd/" \
		-ignore-filename-pattern="kubernetes/manifests/elastic/" \
		-ignore-filename-pattern="kubernetes/manifests/kafka/" \
		-ignore-filename-pattern="kubernetes/manifests/prometheus/" \
		-ignore-filename-pattern="kubernetes/manifests/yugabyte/" \
		$$(git ls-files "kubernetes/manifests/")
lint-protocol-buffers:
	buf lint
lint-c-cpp:
	clang-format -i -style=file **/*.c **/*.cpp **/*.h **/*.ino
lint-terraform:
	terraform fmt -recursive -check
lint-terraform-fix:
	terraform fmt -recursive
