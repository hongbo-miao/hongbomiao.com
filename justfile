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

docker-show-disk-usage:
    docker system df

docker-prune:
    docker system prune

docker-prune-build-cache:
    docker builder prune --all

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

rbenv-install-list-latest-stable-versions:
    rbenv install --list

rbenv-install:
    rbenv install 3.3.6

rbenv-list-versions:
    rbenv versions

rbenv-uninstall:
    rbenv uninstall --force 3.3.6

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

# uv
uv-install-python:
    uv python install

uv-update-lock-file:
    uv lock

uv-install-dependencies:
    uv sync --dev

uv-add:
    uv add xxx

uv-add-dev:
    uv add xxx --dev

# JupyterLab
jupyter-lab:
    jupyter-lab

# Jupyter Notebook
jupyter-notebook-clean:
    uv run poe clean-jupyter-notebook

# SQLFluff
sqlfluff-dialect-list:
    uv run poe sqlfluff-list-dialects

# Rust
rust-install-toolchain:
    rustup install

rust-update:
    rustup update

rust-show-toolchains:
    rustup show

rustup-update:
    rustup self update

cargo-new:
    cargo new xxx

cargo-clean:
    cargo clean

cargo-update-lock-file:
    cargo generate-lockfile

cargo-update:
    cargo update

cargo-check:
    cargo check

cargo-add:
    cargo add xxx

cargo-add-features:
    cargo add xxx --features=xxx

cargo-add-dev:
    cargo add xxx --dev

cargo-add-build:
    cargo add xxx --build

cargo-build-development:
    cargo build

cargo-build-production:
    cargo build --release

cargo-run-development:
    cargo run

cargo-run-production:
    cargo run --release

# Lint
lint-ansible:
    uv run poe lint-ansible

lint-c-cpp-cpplint:
    uv run poe lint-c-cpp-cpplint --repository=asterios/led-blinker --extensions=c,h --recursive asterios/led-blinker
    uv run poe lint-c-cpp-cpplint --repository=compiler-infrastructure/llvm --extensions=cpp,hpp --recursive compiler-infrastructure/llvm
    uv run poe lint-c-cpp-cpplint --repository=data-processing/kafka/kafka-client/kafka-c/avro-producer --extensions=c,h --recursive data-processing/kafka/kafka-client/kafka-c/avro-producer
    uv run poe lint-c-cpp-cpplint --repository=embedded-system/freertos --extensions=ino --recursive embedded-system/freertos
    uv run poe lint-c-cpp-cpplint --repository=matlab/call-c-function-in-matlab --extensions=c,h --recursive matlab/call-c-function-in-matlab
    uv run poe lint-c-cpp-cpplint --repository=matlab/call-c-function-in-matlab --extensions=c,h --recursive matlab/call-c-function-in-matlab
    uv run poe lint-c-cpp-cpplint --repository=parallel-computing/cuda --extensions=cu,cuh --recursive parallel-computing/cuda
    uv run poe lint-c-cpp-cpplint --repository=reverse-engineering/hello-c --extensions=c,h --recursive reverse-engineering/hello-c
    uv run poe lint-c-cpp-cpplint --repository=reverse-engineering/hello-cpp --extensions=cpp,hpp --recursive reverse-engineering/hello-cpp
    uv run poe lint-c-cpp-cpplint --repository=robotics/robot-operating-system/src/hm_cpp_package --extensions=cpp,hpp --recursive robotics/robot-operating-system

lint-c-cpp-clang-format-fix:
    clang-format -i -style=file $(git ls-files "**/*.c" "**/*.cpp" "**/*.cu" "**/*.h" "**/*.ino")

lint-cmake:
    uv run poe lint-cmake

lint-css:
    npm run lint-css

lint-css-fix:
    npm run lint-css-fix

lint-dockerfile:
    hadolint $(git ls-files "**/Dockerfile*")

lint-html:
    npm run lint-html

lint-html-fix:
    npm run lint-html-fix

lint-json:
    npm run lint-json

lint-json-fix:
    npm run lint-json-fix

lint-justfile:
    uv run poe lint-justfile

lint-justfile-fix:
    uv run poe lint-justfile-fix

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
    npm run lint-markdown

lint-markdown-fix:
    npm run lint-markdown-fix

lint-matlab:
    uv run poe lint-matlab

lint-matlab-fix:
    uv run poe lint-matlab-fix

lint-natural-language:
    npm run lint-natural-language

lint-natural-language-fix:
    npm run lint-natural-language-fix

lint-protocol-buffers:
    buf lint

lint-python:
    uv run poe lint-python

lint-python-fix:
    uv run poe lint-python-fix

lint-qml:
    qmllint $(git ls-files "**/*.qml")

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
    shellcheck $(git ls-files "**/*.sh")

lint-solidity:
    npm run lint-solidity

lint-solidity-fix:
    npm run lint-solidity-fix

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

lint-opentofu:
    tofu fmt -recursive -check

lint-opentofu-fix:
    tofu fmt -recursive

lint-toml:
    taplo fmt --check

lint-toml-fix:
    taplo fmt

lint-verilog:
    verible-verilog-lint $(git ls-files "**/*.v") && verible-verilog-format --verify $(git ls-files "**/*.v")

lint-verilog-fix:
    verible-verilog-lint --autofix=inplace $(git ls-files "**/*.v") && verible-verilog-format --inplace $(git ls-files "**/*.v")

lint-vhdl:
    uv run poe lint-vhdl

lint-vhdl-fix:
    uv run poe lint-vhdl-fix

lint-xml:
    npm run lint-xml

lint-xml-fix:
    npm run lint-xml-fix

lint-yaml:
    uv run poe lint-yaml

# Static type check
static-type-check-python:
    uv run poe static-type-check-python --package=aerospace.hm-aerosandbox
    uv run poe static-type-check-python --package=aerospace.hm-openaerostruct
    uv run poe static-type-check-python --package=aerospace.x-plane.rest-api
    uv run poe static-type-check-python --package=aerospace.x-plane.udp
    uv run poe static-type-check-python --package=api-python
    uv run poe static-type-check-python --package=api-rust
    uv run poe static-type-check-python --package=audio.automatic-speech-recognition.hm-faster-whisper
    uv run poe static-type-check-python --package=audio.automatic-speech-recognition.hm-speaches
    uv run poe static-type-check-python --package=audio.automatic-speech-recognition.hm-whisperx
    uv run poe static-type-check-python --package=audio.voice-activity-detector.hm-silero-vad
    uv run poe static-type-check-python --package=audio.voice-activity-detector.hm-webrtcvad
    uv run poe static-type-check-python --package=authorization.hm-opal-client
    uv run poe static-type-check-python --package=cloud-computing.hm-ray.applications.calculate
    uv run poe static-type-check-python --package=cloud-computing.hm-ray.applications.process-flight-data
    uv run poe static-type-check-python --package=cloud-platform.aws.amazon-sagemaker.pytorch-mnist
    uv run poe static-type-check-python --package=cloud-platform.aws.aws-parallelcluster.pcluster
    uv run poe static-type-check-python --package=computer-vision.hm-open3d
    uv run poe static-type-check-python --package=computer-vision.hm-pyvista.mount-saint-helens
    uv run poe static-type-check-python --package=computer-vision.hm-supervision.detect-objects
    uv run poe static-type-check-python --package=data-analytics.hm-cudf.analyze-transactions
    uv run poe static-type-check-python --package=data-analytics.hm-cupy
    uv run poe static-type-check-python --package=data-analytics.hm-daft.analyze-transactions
    uv run poe static-type-check-python --package=data-analytics.hm-geopandas
    uv run poe static-type-check-python --package=data-analytics.hm-marimo
    uv run poe static-type-check-python --package=data-analytics.hm-narwhals
    uv run poe static-type-check-python --package=data-analytics.hm-numba
    uv run poe static-type-check-python --package=data-analytics.hm-pandas.analyze-transactions
    uv run poe static-type-check-python --package=data-analytics.hm-polars.analyze-transactions-cpu
    uv run poe static-type-check-python --package=data-analytics.hm-polars.analyze-transactions-gpu
    uv run poe static-type-check-python --package=data-crawling.hm-crawl4ai
    uv run poe static-type-check-python --package=data-crawling.hm-firecrawl
    uv run poe static-type-check-python --package=data-distribution.arrow-flight.arrow-flight-client
    uv run poe static-type-check-python --package=data-distribution.rti-connext-dds
    uv run poe static-type-check-python --package=data-extraction.hm-docling
    uv run poe static-type-check-python --package=data-extraction.hm-mineru
    uv run poe static-type-check-python --package=data-extraction.hm-olmocr
    uv run poe static-type-check-python --package=data-orchestration.hm-airflow
    uv run poe static-type-check-python --package=data-orchestration.hm-prefect.workflows.calculate
    uv run poe static-type-check-python --package=data-orchestration.hm-prefect.workflows.daft-analysis
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
    uv run poe static-type-check-python --package=data-storage.hm-duckdb.query-duckdb
    uv run poe static-type-check-python --package=data-storage.hm-duckdb.query-lance
    uv run poe static-type-check-python --package=data-storage.hm-duckdb.query-parquet
    uv run poe static-type-check-python --package=data-storage.hm-duckdb.query-protobuf
    uv run poe static-type-check-python --package=data-storage.hm-hdf5
    uv run poe static-type-check-python --package=data-storage.hm-lancedb
    uv run poe static-type-check-python --package=data-storage.hm-protobuf
    uv run poe static-type-check-python --package=data-storage.lance
    uv run poe static-type-check-python --package=data-visualization.grafana.hm-dashboard
    uv run poe static-type-check-python --package=data-visualization.hm-dash.csv-visualizer
    uv run poe static-type-check-python --package=data-visualization.hm-dash.parquet-visualizer
    uv run poe static-type-check-python --package=data-visualization.hm-plotly.heatmap
    uv run poe static-type-check-python --package=data-visualization.hm-pygwalker
    uv run poe static-type-check-python --package=data-visualization.iads.iads-data-manager.iads-config-reader
    uv run poe static-type-check-python --package=data-visualization.iads.iads-data-manager.iads-data-reader
    uv run poe static-type-check-python --package=embedded-system.decode-can-blf-data
    uv run poe static-type-check-python --package=embedded-system.decode-can-mf4-data
    uv run poe static-type-check-python --package=embedded-system.decode-can-trc-data
    uv run poe static-type-check-python --package=embedded-system.format-can-data
    uv run poe static-type-check-python --package=embedded-system.hm-serial
    uv run poe static-type-check-python --package=embedding.hm-imagebind
    uv run poe static-type-check-python --package=embedding.open-clip
    uv run poe static-type-check-python --package=hardware-in-the-loop.national-instruments.hm-pyvisa
    uv run poe static-type-check-python --package=hardware-in-the-loop.national-instruments.hm-tdms
    uv run poe static-type-check-python --package=hardware-in-the-loop.national-instruments.veristand.hm-veristand
    uv run poe static-type-check-python --package=hm-locust
    uv run poe static-type-check-python --package=hm-xxhash
    uv run poe static-type-check-python --package=llm-application.hm-langchain.applications.chat-pdf
    uv run poe static-type-check-python --package=llm-application.hm-langgraph.applications.chat-pdf
    uv run poe static-type-check-python --package=llm-application.hm-llama-index.applications.chat-pdf
    uv run poe static-type-check-python --package=llm-application.hm-pydantic-ai.applications.chat-pdf
    uv run poe static-type-check-python --package=llm-evaluation.hm-deepeval
    uv run poe static-type-check-python --package=llm-inference.hm-mlx-lm
    uv run poe static-type-check-python --package=llm-inference.hm-sglang
    uv run poe static-type-check-python --package=llm-inference.kv-caching
    uv run poe static-type-check-python --package=llm-inference.speculative-decoding
    uv run poe static-type-check-python --package=llm-post-training.fine-tuning.fine-tune-whisper
    uv run poe static-type-check-python --package=llm-post-training.fine-tuning.hm-axolotl
    uv run poe static-type-check-python --package=llm-post-training.fine-tuning.hm-llama-factory
    uv run poe static-type-check-python --package=llm-post-training.fine-tuning.hm-torchtune
    uv run poe static-type-check-python --package=llm-post-training.fine-tuning.hm-unsloth
    uv run poe static-type-check-python --package=llm-post-training.post-training-quantization
    uv run poe static-type-check-python --package=llm-training.automatic-mixed-precision
    uv run poe static-type-check-python --package=machine-learning.convolutional-neural-network
    uv run poe static-type-check-python --package=machine-learning.feature-store
    uv run poe static-type-check-python --package=machine-learning.graph-neural-network
    uv run poe static-type-check-python --package=machine-learning.hm-cuml
    uv run poe static-type-check-python --package=machine-learning.hm-gradio.applications.classify-image
    uv run poe static-type-check-python --package=machine-learning.hm-kubeflow.pipelines.calculate
    uv run poe static-type-check-python --package=machine-learning.hm-kubeflow.pipelines.classify-mnist
    uv run poe static-type-check-python --package=machine-learning.hm-mlflow.experiments.classify-mnist
    uv run poe static-type-check-python --package=machine-learning.hm-mlflow.experiments.predict-diabetes
    uv run poe static-type-check-python --package=machine-learning.hm-nvidia-modulus
    uv run poe static-type-check-python --package=machine-learning.hm-scikit-learn
    uv run poe static-type-check-python --package=machine-learning.hm-streamlit.applications.live-line-chart
    uv run poe static-type-check-python --package=machine-learning.hm-streamlit.applications.map
    uv run poe static-type-check-python --package=machine-learning.hugging-face
    uv run poe static-type-check-python --package=machine-learning.hyperparameter-optimization
    uv run poe static-type-check-python --package=machine-learning.model-context-protocol
    uv run poe static-type-check-python --package=machine-learning.neural-forecasting.forecast-air-passenger-number
    uv run poe static-type-check-python --package=machine-learning.nvidia-dali
    uv run poe static-type-check-python --package=machine-learning.nvidia-triton-inference-server.amazon-sagemaker-triton-resnet-50.deploy
    uv run poe static-type-check-python --package=machine-learning.nvidia-triton-inference-server.amazon-sagemaker-triton-resnet-50.infer
    uv run poe static-type-check-python --package=machine-learning.reinforcement-learning.cart-pole
    uv run poe static-type-check-python --package=machine-learning.stable-diffusion
    uv run poe static-type-check-python --package=machine-learning.transformer
    uv run poe static-type-check-python --package=parallel-computing.hm-taichi.count-primes
    uv run poe static-type-check-python --package=parallel-computing.hm-taichi.fluid-solver
    uv run poe static-type-check-python --package=parallel-computing.hm-triton
    uv run poe static-type-check-python --package=physics.hm-genesis
    uv run poe static-type-check-python --package=quantum-computing
    uv run poe static-type-check-python --package=scientific-computing.hm-sunpy
    uv run poe static-type-check-python --package=scientific-computing.surrogate-model
    uv run poe static-type-check-python --package=tokenization.byte-pair-encoding

static-type-check-terraform:
    cd cloud-infrastructure/opentofu/environments/production/airbyte && tofu validate
    cd cloud-infrastructure/opentofu/environments/production/aws/data && tofu validate
    cd cloud-infrastructure/opentofu/environments/production/aws/general && tofu validate
    cd cloud-infrastructure/opentofu/environments/production/aws/kubernetes && tofu validate
    cd cloud-infrastructure/opentofu/environments/production/aws/network && tofu validate
    cd cloud-infrastructure/opentofu/environments/production/harbor && tofu validate
    cd cloud-infrastructure/opentofu/environments/production/snowflake/account && tofu validate
    cd cloud-infrastructure/opentofu/environments/production/snowflake/data && tofu validate
    cd cloud-infrastructure/opentofu/environments/production/snowflake/general && tofu validate

static-type-check-typescript:
    cd api-node && npm run static-type-check-typescript
    cd ethereum && npm run static-type-check-typescript
    cd data-visualization/grafana/hm-panel-plugin && npm run static-type-check-typescript
    cd mobile/mobile-react-native && npm run static-type-check-typescript
    cd web && npm run static-type-check-typescript
    cd web-cypress && npm run static-type-check-typescript
