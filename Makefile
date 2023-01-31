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

# Git
git-branch-switch:
	git switch xxx
git-branch-create-and-switch:
	git switch -c xxx

git-branch-list-local:
	git branch
git-branch-list-remote:
	git branch -r

git-fetch-branches:
	git fetch --all
git-pull-rebase:
	git pull --rebase

git-log:
	git log

# Docker
docker-login:
	docker login
docker-sh:
	docker run --interactive --tty --rm hm-graphql-server sh
docker-ps:
	docker ps
docker-ps-all:
	docker ps --all
docker-stop:
	docker stop xxx
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
conda-create:
	conda create --name=xxx python=3.10 --yes
conda-env-remove:
	conda env remove --name=xxx
conda-activate:
	conda activate xxx
conda-deactivate:
	conda deactivate
conda-env-list:
	conda env list
conda-list-packages:
	conda list

pyenv-list-versions:
	pyenv versions
pyenv-install:
	pyenv install 3.11
pyenv-local:
	pyenv local 3.11
pyenv-deactivate:
	pyenv shell system

poetry-install-on-macos:
	brew install poetry
poetry-install-on-linux:
	curl -sSL https://install.python-poetry.org | python3 -
poetry-self-update:
	poetry self update
poetry-env-list:
	# ~/Library/Caches/pypoetry/virtualenvs
	poetry env list
poetry-env-remove:
	poetry env remove xxx
poetry-env-use:
	poetry env use 3.11
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev
poetry-shell:
	poetry shell
poetry-check:
	poetry check
poetry-cache-clear:
	poetry cache clear pypi --all

lint-cmake:
	poetry run poe lint-cmake
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
lint-sql:
	poetry run poe lint-sql -- --dialect=bigquery bigquery-ml
	poetry run poe lint-sql -- --dialect=postgres hasura-graphql-engine/migrations
	poetry run poe lint-sql -- --dialect=postgres hasura-graphql-engine/seeds
	poetry run poe lint-sql -- --dialect=postgres kubernetes/data/postgres/opa_db/migrations
	poetry run poe lint-sql -- --dialect=postgres streaming/migrations
	poetry run poe lint-sql -- --dialect=postgres timescaledb/migrations
lint-sql-fix:
	poetry run poe lint-sql-fix -- --dialect=bigquery bigquery-ml
	poetry run poe lint-sql-fix -- --dialect=postgres hasura-graphql-engine/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres hasura-graphql-engine/seeds
	poetry run poe lint-sql-fix -- --dialect=postgres kubernetes/data/postgres/opa_db/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres streaming/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres timescaledb/migrations
lint-yaml:
	poetry run poe lint-yaml
static-type-check-python:
	poetry run poe static-type-check-python -- --package=api-python
	poetry run poe static-type-check-python -- --package=chatbot
	poetry run poe static-type-check-python -- --package=convolutional-neural-network
	poetry run poe static-type-check-python -- --package=data-distribution-service
	poetry run poe static-type-check-python -- --package=feature-store
	poetry run poe static-type-check-python -- --package=grafana.hm-dashboard
	poetry run poe static-type-check-python -- --package=graph-neural-network
	poetry run poe static-type-check-python -- --package=hm-airflow
	poetry run poe static-type-check-python -- --package=hm-locust
	poetry run poe static-type-check-python -- --package=hm-opal-client
	poetry run poe static-type-check-python -- --package=hm-open3d
	poetry run poe static-type-check-python -- --package=hm-prefect
	poetry run poe static-type-check-python -- --package=hm-pyspark
	poetry run poe static-type-check-python -- --package=hugging-face
	poetry run poe static-type-check-python -- --package=quantum-computing

# Lint
lint-dockerfile:
	hadolint $$(git ls-files "*Dockerfile*")
lint-shell:
	shellcheck $$(git ls-files "*.sh")
lint-kubernetes:
	kubeconform \
		-kubernetes-version=1.26.0 \
		-ignore-filename-pattern='.*trafficsplit.yaml' \
		-ignore-filename-pattern='.*my-values.yaml' \
		-ignore-filename-pattern=kubernetes/manifests/argocd/ \
		-ignore-filename-pattern=kubernetes/manifests/elastic/ \
		-ignore-filename-pattern=kubernetes/manifests/kafka/ \
		-ignore-filename-pattern=kubernetes/manifests/prometheus/ \
		-ignore-filename-pattern=kubernetes/manifests/yugabyte/ \
		kubernetes/manifests/
lint-protocol-buffers:
	buf lint
lint-c-cpp:
	clang-format -i -style=file $$(git ls-files "*.c" "*.cpp" "*.h" "*.ino")
lint-qml:
	qmllint $$(git ls-files "*.qml")
lint-terraform:
	terraform fmt -recursive -check
lint-terraform-fix:
	terraform fmt -recursive
