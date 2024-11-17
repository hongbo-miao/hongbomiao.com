# Kubernetes
kubernetes-set-up:
	bash kubernetes/bin/set_up.sh
kubernetes-clean:
	bash kubernetes/bin/clean.sh

# Local
local-set-up:
	bash bin/set_up.sh
local-build:
	bash bin/build.sh
local-clean:
	bash bin/clean.sh

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
	docker compose --file=docker-compose.development.yaml build
	docker compose --file=docker-compose.cypress.yaml build
docker-compose-up:
	docker compose --file=docker-compose.development.yaml up --detach
	docker compose --file=docker-compose.cypress.yaml up --detach
docker-compose-stop:
	docker compose --file=docker-compose.development.yaml stop
	docker compose --file=docker-compose.cypress.yaml stop
docker-compose-down:
	docker compose --file=docker-compose.development.yaml down --volumes
	docker compose --file=docker-compose.cypress.yaml down --volumes

# Node.js
nvm-install:
	nvm install xxx
nvm-uninstall:
	nvm uninstall xxx
nvm-use:
	nvm ls xxx
nvm-alias-default:
	nvm alias default xxx

# rsvg-convert
convert-svg-to-png:
	rsvg-convert Architecture.svg > Architecture.png

# Ruby
ruby-build-upgrade:
	brew upgrade ruby-build
rbenv-list-latest-stable-versions:
	rbenv install --list
rbenv-install:
	rbenv install 3.3.4

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

# .NET
dotnet-sdk-install:
	winget install Microsoft.DotNet.SDK.8

dotnet-tool-initialize:
	dotnet new tool-manifest
dotnet-tool-restore:
	dotnet tool restore
dotnet-tool-install:
	dotnet tool install csharpier
dotnet-tool-list:
	dotnet tool list

# Python
conda-create:
	conda create --name=xxx python=3.12 --yes
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
	pyenv install 3.13
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
poetry-config-show:
	poetry config virtualenvs.in-project
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
	poetry env use python3.13
poetry-env-remove:
	poetry env remove xxx
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-install-no-root:
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

# uv
uv-install-python::
	uv python install
uv-update-lock-file:
	uv lock
uv-install-dependencies:
	uv sync --dev
uv-add:
	uv add xxx
uv-add-dev:
	uv add xxx --dev

# JupterLab
jupyter-lab:
	jupyter-lab

# Jupyter Notebook
jupyter-notebook-clean:
	uv run poe clean-jupyter-notebook cloud-platform/aws/amazon-emr/hm-amazon-emr-cluster-sedona/studio/hm-studio/notebook.ipynb
	uv run poe clean-jupyter-notebook cloud-platform/aws/amazon-sagemaker/pytorch-mnist/notebook.ipynb
	uv run poe clean-jupyter-notebook machine-learning/triton/amazon-sagamaker-triton-resnet-50/set_up/notebook.ipynb

# SQLFluff
sqlfluff-dialect-list:
	uv run poe sqlfluff-list-dialects

# Rust
rust-update:
	rustup update
rust-version:
	rustc --version

rustup-set-default-rust-version:
	rustup default 1.80.1
rustup-update:
	rustup self update
rustup-add:
	rustup component add xxx
rustup-remove:
	rustup component remove xxx

cargo-new:
	cargo new xxx
cargo-update-lock-file:
	cargo generate-lockfile
cargo-update:
	cargo update
cargo-add:
	cargo add xxx
cargo-add-features:
	cargo add xxx --features=xxx
cargo-add-dev:
	cargo add xxx --dev
cargo-add-build:
	cargo add xxx --build

# Lint
lint-ansible:
	uv run poe lint-ansible
lint-c-cpp-fix:
	clang-format -i -style=file $$(git ls-files "**/*.c" "**/*.cpp" "**/*.cu" "**/*.h" "**/*.ino")
lint-cmake:
	uv run poe lint-cmake
lint-css:
	npm run lint:css
lint-css-fix:
	npm run lint:css:fix
lint-dockerfile:
	hadolint $$(git ls-files "**/Dockerfile*")
lint-javascript:
	npm run lint:javascript
lint-javascript-fix:
	npm run lint:javascript:fix
lint-kotlin:
	cd mobile-android && ./gradlew ktlintCheck
lint-kotlin-fix:
	cd mobile-android && ./gradlew ktlintFormat
lint-kubernetes-manifest:
	kubeconform \
		-kubernetes-version=1.26.0 \
		-ignore-filename-pattern='.*trafficsplit.yaml' \
		-ignore-filename-pattern='.*my-values.yaml' \
		-ignore-filename-pattern=kubernetes/manifests/argocd/ \
		-ignore-filename-pattern=kubernetes/manifests/elastic/ \
		-ignore-filename-pattern=kubernetes/manifests/hm-kafka/ \
		-ignore-filename-pattern=kubernetes/manifests/kubeflow/kubeflow-training-operator/ \
		-ignore-filename-pattern=kubernetes/manifests/postgres-operator/ \
		-ignore-filename-pattern=kubernetes/manifests/prometheus/ \
		-ignore-filename-pattern=kubernetes/manifests/yugabyte/ \
		kubernetes/manifests/
lint-markdown:
	npm run lint:markdown
lint-markdown-fix:
	npm run lint:markdown:fix
lint-matlab:
	uv run poe lint-matlab
lint-matlab-fix:
	uv run poe lint-matlab-fix
lint-protocol-buffers:
	buf lint
lint-python-black:
	uv run poe lint-python-black
lint-python-black-fix:
	uv run poe lint-python-black-fix
lint-python-ruff:
	uv run poe lint-python-ruff
lint-python-ruff-fix:
	uv run poe lint-python-ruff-fix
lint-python-isort:
	uv run poe lint-python-isort
lint-python-isort-fix:
	uv run poe lint-python-isort-fix
lint-qml:
	qmllint $$(git ls-files "**/*.qml")
lint-ruby:
	bundle exec rubocop
lint-ruby-fix:
	bundle exec rubocop --autocorrect-all
lint-scala:
	cd data-processing/hm-spark/applications/find-retired-people-scala && sbt scalafmtCheckAll && sbt "scalafixAll --check"
	cd data-processing/hm-spark/applications/ingest-from-s3-to-kafka && sbt scalafmtCheckAll && sbt "scalafixAll --check"
lint-scala-fix:
	cd data-processing/hm-spark/applications/find-retired-people-scala && sbt scalafmtAll && sbt scalafixAll
	cd data-processing/hm-spark/applications/ingest-from-s3-to-kafka && sbt scalafmtAll && sbt scalafixAll
lint-shell:
	shellcheck $$(git ls-files "**/*.sh")
lint-solidity:
	npm run lint:solidity
lint-solidity-fix:
	npm run lint:solidity:fix
lint-sql:
	uv run poe lint-sql --dialect=athena cloud-platform/aws/amazon-athena/queries
	uv run poe lint-sql --dialect=bigquery cloud-platform/google-cloud/bigquery/bigquery-ml
	uv run poe lint-sql --dialect=clickhouse data-storage/clickhouse/cpu_metrics
	uv run poe lint-sql --dialect=postgres hasura-graphql-engine/migrations
	uv run poe lint-sql --dialect=postgres hasura-graphql-engine/seeds
	uv run poe lint-sql --dialect=postgres kubernetes/data/postgres/opa_db/migrations
	uv run poe lint-sql --dialect=postgres data-ingestion/airbyte/sources/postgres/production-iot
	uv run poe lint-sql --dialect=postgres data-processing/flink/applications/stream-tweets/migrations
	uv run poe lint-sql --dialect=postgres data-storage/timescaledb/dummy_iot/migrations
	uv run poe lint-sql --dialect=postgres data-storage/timescaledb/motor/migrations
	uv run poe lint-sql --dialect=postgres ops/argo-cd/applications/production-hm/airbyte/sql
	uv run poe lint-sql --dialect=snowflake data-storage/snowflake/queries
	uv run poe lint-sql --dialect=sparksql data-storage/delta-lake/queries
	uv run poe lint-sql --dialect=sqlite data-storage/sqlite/queries
	uv run poe lint-sql --dialect=trino trino/queries
	uv run poe lint-sql --dialect=tsql data-storage/microsoft-sql-server/queries
lint-sql-fix:
	uv run poe lint-sql-fix --dialect=athena cloud-platform/aws/amazon-athena/queries
	uv run poe lint-sql-fix --dialect=bigquery cloud-platform/google-cloud/bigquery/bigquery-ml
	uv run poe lint-sql-fix --dialect=clickhouse data-storage/clickhouse/cpu_metrics
	uv run poe lint-sql-fix --dialect=postgres hasura-graphql-engine/migrations
	uv run poe lint-sql-fix --dialect=postgres hasura-graphql-engine/seeds
	uv run poe lint-sql-fix --dialect=postgres kubernetes/data/postgres/opa_db/migrations
	uv run poe lint-sql-fix --dialect=postgres data-ingestion/airbyte/sources/postgres/production-iot
	uv run poe lint-sql-fix --dialect=postgres data-processing/flink/applications/stream-tweets/migrations
	uv run poe lint-sql-fix --dialect=postgres data-storage/timescaledb/dummy_iot/migrations
	uv run poe lint-sql-fix --dialect=postgres data-storage/timescaledb/motor/migrations
	uv run poe lint-sql-fix --dialect=postgres ops/argo-cd/applications/production-hm/airbyte/sql
	uv run poe lint-sql-fix --dialect=snowflake data-storage/snowflake/queries
	uv run poe lint-sql-fix --dialect=sparksql queries
	uv run poe lint-sql-fix --dialect=sqlite data-storage/sqlite/queries
	uv run poe lint-sql-fix --dialect=trino trino/queries
	uv run poe lint-sql-fix --dialect=tsql data-storage/microsoft-sql-server/queries
lint-terraform:
	terraform fmt -recursive -check
lint-terraform-fix:
	terraform fmt -recursive
lint-vhdl:
	uv run poe lint-vhdl
lint-vhdl-fix:
	uv run poe lint-vhdl-fix
lint-xml:
	npm run lint:xml
lint-xml-fix:
	npm run lint:xml:fix
lint-yaml:
	uv run poe lint-yaml

# Static type check
static-type-check-python:
	uv run poe static-type-check-python --package=aerospace.hm-aerosandbox
	uv run poe static-type-check-python --package=aerospace.hm-openaerostruct
	uv run poe static-type-check-python --package=api-python
	uv run poe static-type-check-python --package=authorization.hm-opal-client
	uv run poe static-type-check-python --package=cloud-computing.hm-ray.applications.calculate
	uv run poe static-type-check-python --package=cloud-computing.hm-ray.applications.process-flight-data
	uv run poe static-type-check-python --package=cloud-platform.aws.amazon-sagemaker.pytorch-mnist
	uv run poe static-type-check-python --package=computer-vision.hm-open3d
	uv run poe static-type-check-python --package=computer-vision.hm-pyvista.mount-saint-helens
	uv run poe static-type-check-python --package=data-analytics.hm-geopandas
	uv run poe static-type-check-python --package=data-distribution-service
	uv run poe static-type-check-python --package=data-orchestration.hm-airflow
	uv run poe static-type-check-python --package=data-orchestration.hm-prefect.workflows.calculate
	uv run poe static-type-check-python --package=data-orchestration.hm-prefect.workflows.greet
	uv run poe static-type-check-python --package=data-orchestration.hm-prefect.workflows.print-platform
	uv run poe static-type-check-python --package=data-processing.hm-spark.applications.analyze-coffee-customers
	uv run poe static-type-check-python --package=data-processing.hm-spark.applications.find-retired-people-python
	uv run poe static-type-check-python --package=data-processing.hm-spark.applications.find-taxi-top-routes
	uv run poe static-type-check-python --package=data-processing.hm-spark.applications.find-taxi-top-routes-sql
	uv run poe static-type-check-python --package=data-processing.hm-spark.applications.recommend-movies
	uv run poe static-type-check-python --package=data-storage.delta-lake.read-delta-lake-by-amazon-athena
	uv run poe static-type-check-python --package=data-storage.delta-lake.read-delta-lake-by-trino
	uv run poe static-type-check-python --package=data-storage.delta-lake.write-to-delta-lake
	uv run poe static-type-check-python --package=data-visualization.grafana.hm-dashboard
	uv run poe static-type-check-python --package=embedded.decode-can-data
	uv run poe static-type-check-python --package=embedded.format-can-data
	uv run poe static-type-check-python --package=embedded.hm-serial
	uv run poe static-type-check-python --package=hardware-in-the-loop.national-instruments.hm-pyvisa
	uv run poe static-type-check-python --package=hardware-in-the-loop.national-instruments.hm-tdms
	uv run poe static-type-check-python --package=hardware-in-the-loop.national-instruments.hm-ni-veristand
	uv run poe static-type-check-python --package=hm-locust
	uv run poe static-type-check-python --package=hm-xxhash
	uv run poe static-type-check-python --package=machine-learning.convolutional-neural-network
	uv run poe static-type-check-python --package=machine-learning.feature-store
	uv run poe static-type-check-python --package=machine-learning.graph-neural-network
	uv run poe static-type-check-python --package=machine-learning.hm-gradio.applications.classify-image
	uv run poe static-type-check-python --package=machine-learning.hm-kubeflow.pipelines.calculate
	uv run poe static-type-check-python --package=machine-learning.hm-kubeflow.pipelines.classify-mnist
	uv run poe static-type-check-python --package=machine-learning.hm-langchain.applications.chat-pdf
	uv run poe static-type-check-python --package=machine-learning.hm-mlflow.experiments.classify-mnist
	uv run poe static-type-check-python --package=machine-learning.hm-mlflow.experiments.predict-diabetes
	uv run poe static-type-check-python --package=machine-learning.hm-rasa
	uv run poe static-type-check-python --package=machine-learning.hm-streamlit.applications.live-line-chart
	uv run poe static-type-check-python --package=machine-learning.hm-streamlit.applications.map
	uv run poe static-type-check-python --package=machine-learning.hm-supervision.detect-objects
	uv run poe static-type-check-python --package=machine-learning.hugging-face
	uv run poe static-type-check-python --package=machine-learning.neural-forecasting.forecast-air-passenger-number
	uv run poe static-type-check-python --package=machine-learning.reinforcement-learning.cart-pole
	uv run poe static-type-check-python --package=machine-learning.triton.amazon-sagamaker-triton-resnet-50.deploy
	uv run poe static-type-check-python --package=machine-learning.triton.amazon-sagamaker-triton-resnet-50.infer
	uv run poe static-type-check-python --package=quantum-computing
static-type-check-terraform:
	cd cloud-infrastructure/terraform/environments/development/aws && terraform validate
	cd cloud-infrastructure/terraform/environments/production/aws && terraform validate
	cd cloud-infrastructure/terraform/environments/development/snowflake && terraform validate
	cd cloud-infrastructure/terraform/environments/production/snowflake && terraform validate
static-type-check-typescript:
	cd api-node && npm run tsc
	cd ethereum && npm run tsc
	cd mobile-react-native && npm run tsc
	cd web-cypress && npm run tsc
