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

# Git
git-add:
	git add .
git-commit:
	git commit --message="xxx"

git-branch-switch:
	git switch branch-name
git-branch-create-and-switch:
	git switch -c branch-name
git-branch-create-by-commit-hash:
	git branch branch-name xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
git-recover:
	git reflog --no-abbrev
	git branch branch-name xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Git LFS
git-lfs-install:
	git lfs install
git-lfs-track:
	git lfs track "*.mp4"
git-lfs-untrack:
	git lfs untrack "*.mp4"
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

# rsvg-convert
convert-svg-to-png:
	rsvg-convert Architecture.svg > Architecture.png

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
	poetry env use python3.12
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

# Jupyter Notebook
jupyter-notebook-clean:
	poetry run poe clean-jupyter-notebook -- cloud-platform/aws/amazon-emr/hm-amazon-emr-cluster-sedona/studio/hm-studio/main.ipynb
	poetry run poe clean-jupyter-notebook -- cloud-platform/aws/amazon-sagemaker/pytorch-mnist/notebook.ipynb

# SQLFluff
sqlfluff-dialect-list:
	poetry run poe sqlfluff-list-dialects

# Rust
rust-update:
	rustup update
rust-version:
	rustc --version

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
cargo-add-dev:
	cargo add xxx --dev

# Lint
lint-ansible:
	poetry run poe lint-ansible
lint-c-cpp-fix:
	clang-format -i -style=file $$(git ls-files "**/*.c" "**/*.cpp" "**/*.cu" "**/*.h" "**/*.ino")
lint-cmake:
	poetry run poe lint-cmake
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
	poetry run poe lint-matlab
lint-matlab-fix:
	poetry run poe lint-matlab-fix
lint-protocol-buffers:
	buf lint
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
lint-qml:
	qmllint $$(git ls-files "**/*.qml")
lint-ruby:
	bundle exec rubocop
lint-ruby-fix:
	bundle exec rubocop --autocorrect-all
lint-rust-rustfmt:
	cd hm-rust && cargo fmt --all -- --check
lint-rust-rustfmt-fix:
	cd hm-rust && cargo fmt --all
lint-rust-clippy:
	cd hm-rust && cargo clippy
lint-rust-clippy-fix:
	cd hm-rust && cargo clippy --fix --allow-dirty --allow-staged
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
	poetry run poe lint-sql -- --dialect=athena cloud-platform/aws/amazon-athena/queries
	poetry run poe lint-sql -- --dialect=bigquery cloud-platform/google-cloud/bigquery/bigquery-ml
	poetry run poe lint-sql -- --dialect=clickhouse data-storage/clickhouse/cpu_metrics
	poetry run poe lint-sql -- --dialect=postgres hasura-graphql-engine/migrations
	poetry run poe lint-sql -- --dialect=postgres hasura-graphql-engine/seeds
	poetry run poe lint-sql -- --dialect=postgres kubernetes/data/postgres/opa_db/migrations
	poetry run poe lint-sql -- --dialect=postgres data-ingestion/airbyte/postgres
	poetry run poe lint-sql -- --dialect=postgres data-processing/flink/applications/stream-tweets/migrations
	poetry run poe lint-sql -- --dialect=postgres data-storage/timescaledb/dummy_iot/migrations
	poetry run poe lint-sql -- --dialect=postgres data-storage/timescaledb/motor/migrations
	poetry run poe lint-sql -- --dialect=snowflake data-storage/snowflake/queries
	poetry run poe lint-sql -- --dialect=sparksql data-storage/delta-lake/queries
	poetry run poe lint-sql -- --dialect=sqlite data-storage/sqlite/queries
	poetry run poe lint-sql -- --dialect=trino trino/queries
	poetry run poe lint-sql -- --dialect=tsql data-storage/microsoft-sql-server/queries
lint-sql-fix:
	poetry run poe lint-sql-fix -- --dialect=athena cloud-platform/aws/amazon-athena/queries
	poetry run poe lint-sql-fix -- --dialect=bigquery cloud-platform/google-cloud/bigquery/bigquery-ml
	poetry run poe lint-sql-fix -- --dialect=clickhouse data-storage/clickhouse/cpu_metrics
	poetry run poe lint-sql-fix -- --dialect=postgres hasura-graphql-engine/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres hasura-graphql-engine/seeds
	poetry run poe lint-sql-fix -- --dialect=postgres kubernetes/data/postgres/opa_db/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres data-ingestion/airbyte/postgres
	poetry run poe lint-sql-fix -- --dialect=postgres data-processing/flink/applications/stream-tweets/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres data-storage/timescaledb/dummy_iot/migrations
	poetry run poe lint-sql-fix -- --dialect=postgres data-storage/timescaledb/motor/migrations
	poetry run poe lint-sql-fix -- --dialect=snowflake data-storage/snowflake/queries
	poetry run poe lint-sql-fix -- --dialect=sparksql queries
	poetry run poe lint-sql-fix -- --dialect=sqlite data-storage/sqlite/queries
	poetry run poe lint-sql-fix -- --dialect=trino trino/queries
	poetry run poe lint-sql-fix -- --dialect=tsql data-storage/microsoft-sql-server/queries
lint-terraform:
	terraform fmt -recursive -check
lint-terraform-fix:
	terraform fmt -recursive
lint-vhdl:
	poetry run poe lint-vhdl
lint-vhdl-fix:
	poetry run poe lint-vhdl-fix
lint-xml:
	npm run lint:xml
lint-xml-fix:
	npm run lint:xml:fix
lint-yaml:
	poetry run poe lint-yaml

# Static type check
static-type-check-python:
	poetry run poe static-type-check-python -- --package=api-python
	poetry run poe static-type-check-python -- --package=authorization.hm-opal-client
	poetry run poe static-type-check-python -- --package=cloud-computing.hm-ray.applications.greet
	poetry run poe static-type-check-python -- --package=cloud-platform.aws.amazon-sagemaker.pytorch-mnist
	poetry run poe static-type-check-python -- --package=computer-vision.hm-open3d
	poetry run poe static-type-check-python -- --package=computer-vision.hm-pyvista.mount-saint-helens
	poetry run poe static-type-check-python -- --package=data-analytics.hm-geopandas
	poetry run poe static-type-check-python -- --package=data-distribution-service
	poetry run poe static-type-check-python -- --package=data-orchestration.hm-airflow
	poetry run poe static-type-check-python -- --package=data-orchestration.hm-prefect.workflows.calculate
	poetry run poe static-type-check-python -- --package=data-orchestration.hm-prefect.workflows.greet
	poetry run poe static-type-check-python -- --package=data-orchestration.hm-prefect.workflows.ingest-data
	poetry run poe static-type-check-python -- --package=data-orchestration.hm-prefect.workflows.print-platform
	poetry run poe static-type-check-python -- --package=data-processing.hm-spark.applications.analyze-coffee-customers
	poetry run poe static-type-check-python -- --package=data-processing.hm-spark.applications.find-retired-people-python
	poetry run poe static-type-check-python -- --package=data-processing.hm-spark.applications.find-taxi-top-routes
	poetry run poe static-type-check-python -- --package=data-processing.hm-spark.applications.find-taxi-top-routes-sql
	poetry run poe static-type-check-python -- --package=data-processing.hm-spark.applications.recommend-movies
	poetry run poe static-type-check-python -- --package=data-storage.delta-lake.read-delta-lake-by-amazon-athena
	poetry run poe static-type-check-python -- --package=data-storage.delta-lake.read-delta-lake-by-trino
	poetry run poe static-type-check-python -- --package=data-storage.delta-lake.write-to-delta-lake
	poetry run poe static-type-check-python -- --package=data-visualization.grafana.hm-dashboard
	poetry run poe static-type-check-python -- --package=embedded.decode-can-data
	poetry run poe static-type-check-python -- --package=embedded.format-can-data
	poetry run poe static-type-check-python -- --package=embedded.hm-serial
	poetry run poe static-type-check-python -- --package=hardware-in-the-loop.national-instruments.hm-pyvisa
	poetry run poe static-type-check-python -- --package=hardware-in-the-loop.national-instruments.hm-tdms
	poetry run poe static-type-check-python -- --package=hardware-in-the-loop.national-instruments.hm-ni-veristand
	poetry run poe static-type-check-python -- --package=hm-locust
	poetry run poe static-type-check-python -- --package=hm-xxhash
	poetry run poe static-type-check-python -- --package=machine-learning.convolutional-neural-network
	poetry run poe static-type-check-python -- --package=machine-learning.feature-store
	poetry run poe static-type-check-python -- --package=machine-learning.graph-neural-network
	poetry run poe static-type-check-python -- --package=machine-learning.hm-gradio.applications.classify-image
	poetry run poe static-type-check-python -- --package=machine-learning.hm-kubeflow.pipelines.calculate
	poetry run poe static-type-check-python -- --package=machine-learning.hm-kubeflow.pipelines.classify-mnist
	poetry run poe static-type-check-python -- --package=machine-learning.hm-langchain.applications.chat-pdf
	poetry run poe static-type-check-python -- --package=machine-learning.hm-mlflow.experiments.classify-mnist
	poetry run poe static-type-check-python -- --package=machine-learning.hm-mlflow.experiments.predict-diabetes
	poetry run poe static-type-check-python -- --package=machine-learning.hm-rasa
	poetry run poe static-type-check-python -- --package=machine-learning.hm-streamlit.applications.live-line-chart
	poetry run poe static-type-check-python -- --package=machine-learning.hm-streamlit.applications.map
	poetry run poe static-type-check-python -- --package=machine-learning.hm-supervision.detect-objects
	poetry run poe static-type-check-python -- --package=machine-learning.hugging-face
	poetry run poe static-type-check-python -- --package=machine-learning.neural-forecasting.forecast-air-passenger-number
	poetry run poe static-type-check-python -- --package=machine-learning.reinforcement-learning.cart-pole
	poetry run poe static-type-check-python -- --package=quantum-computing
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
