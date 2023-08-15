# Kubernetes
kubernetes-set-up:
	sh kubernetes/bin/set_up.sh
kubernetes-clean:
	sh kubernetes/bin/clean.sh

# Local
local-set-up:
	sh bin/set_up.sh
local-build:
	sh bin/build.sh
local-clean:
	sh bin/clean.sh

# Git
git-add:
	git add .
git-commit:
	git commit --message="xxx"

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
git-status:
	git status

# Git LFS
git-lfs-install:
	git lfs install
git-lfs-track:
	git lfs track "*.slx"
git-lfs-status:
	git lfs status
git-lfs-list:
	git lfs ls-files

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

# Node.js
nvm-install:
	nvm install xxx
nvm-uninstall:
	nvm uninstall xxx
nvm-use:
	nvm ls xxx
nvm-alias-default:
	nvm alias default xxx

# JupterLab
jupyter-lab:
	jupyter-lab

# Gitleaks
gitleaks-install:
	brew install gitleaks
gitleaks-detect:
	gitleaks detect --source=. --verbose

# Ruby
rbenv-list-latest-stable-versions:
	rbenv install -l
rbenv-install:
	rbenv install 3.2.1

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
pyenv-uninstall:
	pyenv uninstall 3.11
pyenv-local:
	pyenv local 3.11
pyenv-global:
	pyenv global 3.11 3.10 3.8
pyenv-deactivate:
	pyenv shell system

poetry-config-list:
	poetry config --list
poetry-config-set:
	poetry config virtualenvs.in-project true
poetry-self-update:
	poetry self update
poetry-version:
	poetry --version
poetry-env-list:
	# ~/Library/Caches/pypoetry/virtualenvs
	poetry env list
poetry-env-use:
	poetry env use 3.11
poetry-env-remove:
	poetry env remove xxx
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root
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

clean-jupyter-notebook:
	poetry run poe clean-jupyter-notebook -- aws/amazon-emr/studio/hm-studio/hm-workspace.ipynb
	poetry run poe clean-jupyter-notebook -- aws/amazon-sagemaker/pytorch-mnist/notebook.ipynb
lint-ansible:
	poetry run poe lint-ansible
lint-matlab:
	poetry run poe lint-matlab
lint-matlab-fix:
	poetry run poe lint-matlab-fix
lint-cmake:
	poetry run poe lint-cmake
lint-python-black:
	poetry run poe lint-python-black
lint-python-black-fix:
	poetry run poe lint-python-black-fix
lint-python-ruff:
	poetry run poe lint-python-ruff
lint-python-ruff-fix:
	poetry run poe lint-python-ruff-fix
lint-python-isort:
	poetry run poe lint-python-isort
lint-python-isort-fix:
	poetry run poe lint-python-isort-fix
lint-sql:
	poetry run poe lint-sql -- --dialect=bigquery google-cloud/bigquery/bigquery-ml
	poetry run poe lint-sql -- --dialect=clickhouse clickhouse/cpu_metrics
	poetry run poe lint-sql -- --dialect=postgres hasura-graphql-engine/migrations
	poetry run poe lint-sql -- --dialect=postgres hasura-graphql-engine/seeds
	poetry run poe lint-sql -- --dialect=postgres kubernetes/data/postgres/opa_db/migrations
	poetry run poe lint-sql -- --dialect=postgres streaming/migrations
	poetry run poe lint-sql -- --dialect=postgres timescaledb/dummy_iot/migrations
	poetry run poe lint-sql -- --dialect=postgres timescaledb/motor/migrations
	poetry run poe lint-sql -- --dialect=sparksql delta-lake/sql
	poetry run poe lint-sql -- --dialect=sparksql hm-spark/applications/find-taxi-top-routes-sql/src/queries
lint-sql-fix:
	poetry run poe lint-sql-fix -- --dialect=bigquery google-cloud/bigquery/bigquery-ml
	poetry run poe lint-sql-fix -- --dialect=clickhouse clickhouse/cpu_metrics
	poetry run poe lint-sql-fix -- --dialect=postgres hasura-graphql-engine/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres hasura-graphql-engine/seeds
	poetry run poe lint-sql-fix -- --dialect=postgres kubernetes/data/postgres/opa_db/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres streaming/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres timescaledb/dummy_iot/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres timescaledb/motor/migrations
	poetry run poe lint-sql-fix -- --dialect=sparksql delta-lake/sql
	poetry run poe lint-sql-fix -- --dialect=sparksql hm-spark/applications/find-taxi-top-routes-sql/src/queries
lint-vhdl:
	poetry run poe lint-vhdl
lint-vhdl-fix:
	poetry run poe lint-vhdl-fix
lint-yaml:
	poetry run poe lint-yaml
static-type-check-python:
	poetry run poe static-type-check-python -- --package=api-python
	poetry run poe static-type-check-python -- --package=aws.amazon-sagemaker.pytorch-mnist
	poetry run poe static-type-check-python -- --package=chatbot
	poetry run poe static-type-check-python -- --package=convolutional-neural-network
	poetry run poe static-type-check-python -- --package=data-distribution-service
	poetry run poe static-type-check-python -- --package=delta-lake.read-delta-lake-by-amazon-athena
	poetry run poe static-type-check-python -- --package=delta-lake.read-delta-lake-by-trino
	poetry run poe static-type-check-python -- --package=delta-lake.write-to-delta-lake
	poetry run poe static-type-check-python -- --package=feature-store
	poetry run poe static-type-check-python -- --package=grafana.hm-dashboard
	poetry run poe static-type-check-python -- --package=graph-neural-network
	poetry run poe static-type-check-python -- --package=hm-airflow
	poetry run poe static-type-check-python -- --package=hm-kubeflow.pipelines.calculate
	poetry run poe static-type-check-python -- --package=hm-kubeflow.pipelines.classify-mnist
	poetry run poe static-type-check-python -- --package=hm-locust
	poetry run poe static-type-check-python -- --package=hm-mlflow.experiments.classify-mnist
	poetry run poe static-type-check-python -- --package=hm-mlflow.experiments.predict-diabetes
	poetry run poe static-type-check-python -- --package=hm-ni-veristand
	poetry run poe static-type-check-python -- --package=hm-opal-client
	poetry run poe static-type-check-python -- --package=hm-open3d
	poetry run poe static-type-check-python -- --package=hm-prefect.workflows.calculate
	poetry run poe static-type-check-python -- --package=hm-prefect.workflows.greet
	poetry run poe static-type-check-python -- --package=hm-prefect.workflows.ingest-data
	poetry run poe static-type-check-python -- --package=hm-prefect.workflows.print-platform
	poetry run poe static-type-check-python -- --package=hm-ray.applications.greet
	poetry run poe static-type-check-python -- --package=hm-spark.applications.find-retired-people-python
	poetry run poe static-type-check-python -- --package=hm-spark.applications.find-taxi-top-routes
	poetry run poe static-type-check-python -- --package=hm-spark.applications.find-taxi-top-routes-sql
	poetry run poe static-type-check-python -- --package=hm-spark.applications.recommend-movies
	poetry run poe static-type-check-python -- --package=hugging-face
	poetry run poe static-type-check-python -- --package=quantum-computing
	poetry run poe static-type-check-python -- --package=tdms

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
		-ignore-filename-pattern=kubernetes/manifests/hm-kafka/ \
		-ignore-filename-pattern=kubernetes/manifests/kubeflow/kubeflow-training-operator/ \
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
