# Docker
docker-sign-in:
    docker login

docker-sh:
    docker run --rm --interactive --tty hm-graphql-server sh

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

# rsvg-convert
convert-svg-to-png:
    rsvg-convert Architecture.svg > Architecture.png

# Ruby
ruby-build-upgrade:
    brew upgrade ruby-build

rbenv-install-list-latest-stable-versions:
    rbenv install --list

rbenv-install:
    rbenv install 4.0.0

rbenv-list-versions:
    rbenv versions

rbenv-uninstall:
    rbenv uninstall --force 4.0.0

bundle-initialize:
    bundle init

bundle-install:
    bundle install

bundle-add:
    bundle add xxx

bundle-update:
    bundle update

bundle-add-platform:
    bundle lock --add-platform aarch64-linux

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

# sqruff
sqruff-list-dialects:
    uv run poe sqruff-list-dialects

# Lint
lint-ansible:
    cd infrastructure/ansible && just lint-ansible

lint-c-cpp-cpplint:
    uv run poe lint-c-cpp-cpplint --repository=compiler-infrastructure/llvm --extensions=cpp,hpp --recursive compiler-infrastructure/llvm
    uv run poe lint-c-cpp-cpplint --repository=data-processing/kafka/kafka-client/kafka-c/avro-producer --extensions=c,h --recursive data-processing/kafka/kafka-client/kafka-c/avro-producer
    uv run poe lint-c-cpp-cpplint --repository=embedded-system/asterios/led-blinker --extensions=c,h --recursive embedded-system/asterios/led-blinker
    uv run poe lint-c-cpp-cpplint --repository=embedded-system/freertos --extensions=ino --recursive embedded-system/freertos
    uv run poe lint-c-cpp-cpplint --repository=matlab/call-c-function-in-matlab --extensions=c,h --recursive matlab/call-c-function-in-matlab
    uv run poe lint-c-cpp-cpplint --repository=matlab/call-c-function-in-matlab --extensions=c,h --recursive matlab/call-c-function-in-matlab
    uv run poe lint-c-cpp-cpplint --repository=parallel-computing/cuda --extensions=cu,cuh --recursive parallel-computing/cuda
    uv run poe lint-c-cpp-cpplint --repository=reverse-engineering/hello-c --extensions=c,h --recursive reverse-engineering/hello-c
    uv run poe lint-c-cpp-cpplint --repository=reverse-engineering/hello-cpp --extensions=cpp,hpp --recursive reverse-engineering/hello-cpp
    uv run poe lint-c-cpp-cpplint --repository=robotics/robot-operating-system/src/hm_cpp_package --extensions=cpp,hpp --recursive robotics/robot-operating-system

lint-c-cpp-clang-format-fix:
    clang-format -i -style=file $(git ls-files '**/*.c' '**/*.cpp' '**/*.cu' '**/*.h' '**/*.ino')

lint-cmake:
    uv run poe lint-cmake

lint-css:
    npm run lint-css

lint-css-fix:
    npm run lint-css-fix

lint-docker-compose:
    npm run lint-docker-compose

lint-docker-compose-fix:
    npm run lint-docker-compose-fix

lint-dockerfile:
    hadolint $(git ls-files '**/Dockerfile*')

lint-dotenv:
    dotenv-linter check $(git ls-files '**/.env*')

lint-dotenv-fix:
    dotenv-linter fix --no-backup $(git ls-files '**/.env*')

lint-editorconfig:
    uv run poe lint-editorconfig

lint-html:
    npm run lint-html

lint-html-fix:
    npm run lint-html-fix

lint-json-eslint:
    npm run lint-json-eslint

lint-json-eslint-fix:
    npm run lint-json-eslint-fix

lint-json-prettier:
    npm run lint-json-prettier

lint-json-prettier-fix:
    npm run lint-json-prettier-fix

lint-justfile:
    uv run poe lint-justfile

lint-justfile-fix:
    uv run poe lint-justfile-fix

lint-kotlin:
    cd mobile/mobile-android && ./gradlew ktlintCheck

lint-kotlin-fix:
    cd mobile/mobile-android && ./gradlew ktlintFormat

lint-kubernetes-manifest:
    kubeconform \
        -kubernetes-version=1.34.0 \
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
    qmllint $(git ls-files '**/*.qml')

lint-ruby:
    bundle exec rubocop

lint-ruby-fix:
    bundle exec rubocop --autocorrect-all

lint-rust-rustfmt:
    cd api/api-rust && just lint-rust-rustfmt
    cd data-processing/kafka/kafka-client/kafka-rust/proto-consumer && just lint-rust-rustfmt
    cd data-processing/kafka/kafka-client/kafka-rust/proto-consumer && just lint-rust-rustfmt
    cd data-processing/kafka/kafka-client/kafka-rust/proto-producer && just lint-rust-rustfmt
    cd data-processing/kafka/kafka-client/kafka-rust/udp-kafka-bridge && just lint-rust-rustfmt
    cd data-processing/kafka/kafka-client/kafka-rust/zeromq-kafka-bridge && just lint-rust-rustfmt
    cd data-processing/nats/audio-stream-publisher && just lint-rust-rustfmt
    cd data-processing/nats/audio-stream-transcriber && just lint-rust-rustfmt
    cd data-processing/nats/nats-postgres-bridge && just lint-rust-rustfmt
    cd data-transport/arrow-flight/arrow-flight-server && just lint-rust-rustfmt
    cd data-transport/dust-dds/dust-dds-publisher && just lint-rust-rustfmt
    cd data-transport/dust-dds/dust-dds-subscriber && just lint-rust-rustfmt
    cd data-visualization/iads/iads-rtstation/iads-data-producer && just lint-rust-rustfmt
    cd data-visualization/iads/iads-rtstation/zeromq-iads-bridge && just lint-rust-rustfmt
    cd network/udp/udp-receiver && just lint-rust-rustfmt
    cd network/udp/udp-sender && just lint-rust-rustfmt
    cd operating-system/windows/calculator && just lint-rust-rustfmt

lint-rust-rustfmt-fix:
    cd api/api-rust && just lint-rust-rustfmt-fix
    cd data-processing/kafka/kafka-client/kafka-rust/proto-consumer && just lint-rust-rustfmt-fix
    cd data-processing/kafka/kafka-client/kafka-rust/proto-consumer && just lint-rust-rustfmt-fix
    cd data-processing/kafka/kafka-client/kafka-rust/proto-producer && just lint-rust-rustfmt-fix
    cd data-processing/kafka/kafka-client/kafka-rust/udp-kafka-bridge && just lint-rust-rustfmt-fix
    cd data-processing/kafka/kafka-client/kafka-rust/zeromq-kafka-bridge && just lint-rust-rustfmt-fix
    cd data-processing/nats/audio-stream-publisher && just lint-rust-rustfmt-fix
    cd data-processing/nats/audio-stream-transcriber && just lint-rust-rustfmt-fix
    cd data-processing/nats/nats-postgres-bridge && just lint-rust-rustfmt-fix
    cd data-transport/arrow-flight/arrow-flight-server && just lint-rust-rustfmt-fix
    cd data-transport/dust-dds/dust-dds-publisher && just lint-rust-rustfmt-fix
    cd data-transport/dust-dds/dust-dds-subscriber && just lint-rust-rustfmt-fix
    cd data-visualization/iads/iads-rtstation/iads-data-producer && just lint-rust-rustfmt-fix
    cd data-visualization/iads/iads-rtstation/zeromq-iads-bridge && just lint-rust-rustfmt-fix
    cd network/udp/udp-receiver && just lint-rust-rustfmt-fix
    cd network/udp/udp-sender && just lint-rust-rustfmt-fix
    cd operating-system/windows/calculator && just lint-rust-rustfmt-fix

lint-rust-clippy:
    cd api/api-rust && just lint-rust-clippy
    cd data-processing/kafka/kafka-client/kafka-rust/proto-consumer && just lint-rust-clippy
    cd data-processing/kafka/kafka-client/kafka-rust/proto-consumer && just lint-rust-clippy
    cd data-processing/kafka/kafka-client/kafka-rust/proto-producer && just lint-rust-clippy
    cd data-processing/kafka/kafka-client/kafka-rust/udp-kafka-bridge && just lint-rust-clippy
    cd data-processing/kafka/kafka-client/kafka-rust/zeromq-kafka-bridge && just lint-rust-clippy
    cd data-processing/nats/audio-stream-publisher && just lint-rust-clippy
    cd data-processing/nats/audio-stream-transcriber && just lint-rust-clippy
    cd data-processing/nats/nats-postgres-bridge && just lint-rust-clippy
    cd data-transport/arrow-flight/arrow-flight-server && just lint-rust-clippy
    cd data-transport/dust-dds/dust-dds-publisher && just lint-rust-clippy
    cd data-transport/dust-dds/dust-dds-subscriber && just lint-rust-clippy
    cd data-visualization/iads/iads-rtstation/iads-data-producer && just lint-rust-clippy
    cd data-visualization/iads/iads-rtstation/zeromq-iads-bridge && just lint-rust-clippy
    cd network/udp/udp-receiver && just lint-rust-clippy
    cd network/udp/udp-sender && just lint-rust-clippy
    cd operating-system/windows/calculator && just lint-rust-clippy

lint-rust-clippy-fix:
    cd api/api-rust && just lint-rust-clippy-fix
    cd data-processing/kafka/kafka-client/kafka-rust/proto-consumer && just lint-rust-clippy-fix
    cd data-processing/kafka/kafka-client/kafka-rust/proto-consumer && just lint-rust-clippy-fix
    cd data-processing/kafka/kafka-client/kafka-rust/proto-producer && just lint-rust-clippy-fix
    cd data-processing/kafka/kafka-client/kafka-rust/udp-kafka-bridge && just lint-rust-clippy-fix
    cd data-processing/kafka/kafka-client/kafka-rust/zeromq-kafka-bridge && just lint-rust-clippy-fix
    cd data-processing/nats/audio-stream-publisher && just lint-rust-clippy-fix
    cd data-processing/nats/audio-stream-transcriber && just lint-rust-clippy-fix
    cd data-processing/nats/nats-postgres-bridge && just lint-rust-clippy-fix
    cd data-transport/arrow-flight/arrow-flight-server && just lint-rust-clippy-fix
    cd data-transport/dust-dds/dust-dds-publisher && just lint-rust-clippy-fix
    cd data-transport/dust-dds/dust-dds-subscriber && just lint-rust-clippy-fix
    cd data-visualization/iads/iads-rtstation/iads-data-producer && just lint-rust-clippy-fix
    cd data-visualization/iads/iads-rtstation/zeromq-iads-bridge && just lint-rust-clippy-fix
    cd network/udp/udp-receiver && just lint-rust-clippy-fix
    cd network/udp/udp-sender && just lint-rust-clippy-fix
    cd operating-system/windows/calculator && just lint-rust-clippy-fix

lint-scala:
    cd data-processing/hm-spark/applications/find-retired-people-scala && just lint-scala
    cd data-processing/hm-spark/applications/ingest-from-s3-to-kafka && just lint-scala

lint-scala-fix:
    cd data-processing/hm-spark/applications/find-retired-people-scala && just lint-scala-fix
    cd data-processing/hm-spark/applications/ingest-from-s3-to-kafka && just lint-scala-fix

lint-shell:
    shellcheck $(git ls-files '**/*.sh')

lint-solidity:
    npm run lint-solidity

lint-solidity-fix:
    npm run lint-solidity-fix

lint-sql:
    uv run poe lint-sql --dialect=athena cloud-platform/aws/amazon-athena/queries
    uv run poe lint-sql --dialect=bigquery cloud-platform/google-cloud/bigquery/bigquery-ml/detect_sales_anomalies
    uv run poe lint-sql --dialect=bigquery cloud-platform/google-cloud/bigquery/bigquery-ml/predict_taxi_fare
    uv run poe lint-sql --dialect=clickhouse data-storage/clickhouse/cpu_metrics
    uv run poe lint-sql --dialect=postgres api/api-rust/migrations
    uv run poe lint-sql --dialect=postgres api/hasura-graphql-engine/migrations
    uv run poe lint-sql --dialect=postgres api/hasura-graphql-engine/seeds
    uv run poe lint-sql --dialect=postgres data-ingestion/airbyte/sources/postgres/production-iot
    uv run poe lint-sql --dialect=postgres data-processing/flink/applications/stream-tweets/migrations
    uv run poe lint-sql --dialect=postgres data-storage/hm-pgbackrest/sql
    uv run poe lint-sql --dialect=postgres data-storage/timescaledb/dummy_iot/migrations
    uv run poe lint-sql --dialect=postgres data-storage/timescaledb/motor/migrations
    uv run poe lint-sql --dialect=postgres kubernetes/data/postgres/opa_db/migrations
    uv run poe lint-sql --dialect=snowflake data-storage/snowflake/queries
    uv run poe lint-sql --dialect=sparksql data-storage/delta-lake/queries
    uv run poe lint-sql --dialect=sqlite data-storage/sqlite/queries
    uv run poe lint-sql --dialect=trino data-processing/trino/queries
    uv run poe lint-sql --dialect=tsql data-storage/microsoft-sql-server/queries

lint-sql-fix:
    uv run poe lint-sql-fix --dialect=athena cloud-platform/aws/amazon-athena/queries
    uv run poe lint-sql-fix --dialect=bigquery cloud-platform/google-cloud/bigquery/bigquery-ml/detect_sales_anomalies
    uv run poe lint-sql-fix --dialect=bigquery cloud-platform/google-cloud/bigquery/bigquery-ml/predict_taxi_fare
    uv run poe lint-sql-fix --dialect=clickhouse data-storage/clickhouse/cpu_metrics
    uv run poe lint-sql-fix --dialect=postgres api/api-rust/migrations
    uv run poe lint-sql-fix --dialect=postgres api/hasura-graphql-engine/migrations
    uv run poe lint-sql-fix --dialect=postgres api/hasura-graphql-engine/seeds
    uv run poe lint-sql-fix --dialect=postgres data-ingestion/airbyte/sources/postgres/production-iot
    uv run poe lint-sql-fix --dialect=postgres data-processing/flink/applications/stream-tweets/migrations
    uv run poe lint-sql-fix --dialect=postgres data-storage/hm-pgbackrest/sql
    uv run poe lint-sql-fix --dialect=postgres data-storage/timescaledb/dummy_iot/migrations
    uv run poe lint-sql-fix --dialect=postgres data-storage/timescaledb/motor/migrations
    uv run poe lint-sql-fix --dialect=postgres kubernetes/data/postgres/opa_db/migrations
    uv run poe lint-sql-fix --dialect=snowflake data-storage/snowflake/queries
    uv run poe lint-sql-fix --dialect=sparksql data-storage/delta-lake/queries
    uv run poe lint-sql-fix --dialect=sqlite data-storage/sqlite/queries
    uv run poe lint-sql-fix --dialect=trino data-processing/trino/queries
    uv run poe lint-sql-fix --dialect=tsql data-storage/microsoft-sql-server/queries

lint-opentofu:
    tofu fmt -recursive -check

lint-opentofu-fix:
    tofu fmt -recursive

lint-swift-format:
    cd mobile/mobile-ios && just lint-swift-format

lint-swift-format-fix:
    cd mobile/mobile-ios && just lint-swift-format-fix

lint-swift-lint:
    cd mobile/mobile-ios && just lint-swift-lint

lint-toml:
    taplo fmt --check

lint-toml-fix:
    taplo fmt

lint-verilog:
    verible-verilog-lint $(git ls-files '**/*.v') && verible-verilog-format --verify $(git ls-files '**/*.v')

lint-verilog-fix:
    verible-verilog-lint --autofix=inplace $(git ls-files '**/*.v') && verible-verilog-format --inplace $(git ls-files '**/*.v')

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
static-type-check-opentofu:
    cd infrastructure/opentofu/environments/production/airbyte && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/argo-cd && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/aws/general && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/aws/kubernetes && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/aws/network && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/aws/storage && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/grafana && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/harbor && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/nebius/applications && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/nebius/general && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/nebius/kubernetes && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/nebius/network && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/nebius/storage && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/snowflake/account && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/snowflake/general && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/snowflake/storage && just static-type-check-opentofu
    cd infrastructure/opentofu/environments/production/trino-gateway && just static-type-check-opentofu

static-type-check-python:
    cd aerospace/bluesky && just static-type-check-python
    cd aerospace/flightradar24 && just static-type-check-python
    cd aerospace/hm-aerosandbox && just static-type-check-python
    cd aerospace/hm-openaerostruct && just static-type-check-python
    cd aerospace/x-plane/rest-api && just static-type-check-python
    cd aerospace/x-plane/udp && just static-type-check-python
    cd api/api-python && just static-type-check-python
    cd api/api-rust/scripts && just static-type-check-python
    cd audio/audio-signal-analysis/hm-librosa && just static-type-check-python
    cd audio/automatic-speech-recognition-evaluation/hm-jiwer && just static-type-check-python
    cd audio/automatic-speech-recognition-inference/hm-faster-whisper && just static-type-check-python
    cd audio/automatic-speech-recognition-inference/hm-speaches && just static-type-check-python
    cd audio/automatic-speech-recognition-inference/hm-whisperx && just static-type-check-python
    cd audio/automatic-speech-recognition-inference/nvidia-nemo && just static-type-check-python
    cd audio/speaker-diarization/hm-diart && just static-type-check-python
    cd audio/speaker-diarization/hm-senko && just static-type-check-python
    cd audio/speaker-diarization/streaming-sortformer-diarizer && just static-type-check-python
    cd audio/voice-activity-detection/hm-silero-vad && just static-type-check-python
    cd audio/voice-activity-detection/rnnoise-vad && just static-type-check-python
    cd audio/voice-activity-detection/webrtc-vad && just static-type-check-python
    cd autonomy/camera-radar-fusion && just static-type-check-python
    cd cloud-computing/hm-ray/applications/calculate && just static-type-check-python
    cd cloud-computing/hm-ray/applications/daft-analysis && just static-type-check-python
    cd cloud-computing/hm-ray/applications/process-flight-data && just static-type-check-python
    cd cloud-computing/hm-ray/hm-ray && just static-type-check-python
    cd cloud-computing/hm-skypilot && just static-type-check-python
    cd cloud-platform/aws/amazon-sagemaker/pytorch-mnist && just static-type-check-python
    cd cloud-platform/aws/aws-parallelcluster/pcluster && just static-type-check-python
    cd computer-vision/anyup && just static-type-check-python
    cd computer-vision/export-yolo-to-onnx && just static-type-check-python
    cd computer-vision/hm-open3d && just static-type-check-python
    cd computer-vision/hm-pyvista/mount-saint-helens && just static-type-check-python
    cd computer-vision/hm-supervision/detect-objects && just static-type-check-python
    cd computer-vision/vision-transformer/dinov3 && just static-type-check-python
    cd convolutional-neural-network && just static-type-check-python
    cd data-analytics/hm-cudf/analyze-transactions && just static-type-check-python
    cd data-analytics/hm-cupy && just static-type-check-python
    cd data-analytics/hm-daft/analyze-transactions && just static-type-check-python
    cd data-analytics/hm-geopandas && just static-type-check-python
    cd data-analytics/hm-marimo && just static-type-check-python
    cd data-analytics/hm-narwhals && just static-type-check-python
    cd data-analytics/hm-networkx && just static-type-check-python
    cd data-analytics/hm-numba && just static-type-check-python
    cd data-analytics/hm-pandas/analyze-transactions && just static-type-check-python
    cd data-analytics/hm-polars/analyze-transactions-cpu && just static-type-check-python
    cd data-analytics/hm-polars/analyze-transactions-gpu && just static-type-check-python
    cd data-crawling/hm-crawl4ai && just static-type-check-python
    cd data-crawling/hm-firecrawl && just static-type-check-python
    cd data-extraction/hm-docling && just static-type-check-python
    cd data-extraction/hm-mineru && just static-type-check-python
    cd data-extraction/hm-olmocr && just static-type-check-python
    cd data-orchestration/hm-airflow && just static-type-check-python
    cd data-orchestration/hm-prefect/workflows/calculate && just static-type-check-python
    cd data-orchestration/hm-prefect/workflows/daft-analysis && just static-type-check-python
    cd data-orchestration/hm-prefect/workflows/greet && just static-type-check-python
    cd data-orchestration/hm-prefect/workflows/print-platform && just static-type-check-python
    cd data-processing/hm-cocoindex/flows/embed-markdown-to-postgres && just static-type-check-python
    cd data-processing/hm-pathway/pipelines/aggregate-hourly-events && just static-type-check-python
    cd data-processing/hm-spark/applications/analyze-coffee-customers && just static-type-check-python
    cd data-processing/hm-spark/applications/find-retired-people-python && just static-type-check-python
    cd data-processing/hm-spark/applications/find-taxi-top-routes && just static-type-check-python
    cd data-processing/hm-spark/applications/find-taxi-top-routes-sql && just static-type-check-python
    cd data-processing/hm-spark/applications/recommend-movies && just static-type-check-python
    cd data-processing/ingest-flac-to-parquet && just static-type-check-python
    cd data-processing/nats/audio-file-publisher && just static-type-check-python
    cd data-processing/nats/audio-file-subscriber && just static-type-check-python
    cd data-processing/nats/telemetry-stream-publisher && just static-type-check-python
    cd data-processing/nats/telemetry-stream-subscriber && just static-type-check-python
    cd data-storage/delta-lake/read-delta-lake-by-amazon-athena && just static-type-check-python
    cd data-storage/delta-lake/read-delta-lake-by-trino && just static-type-check-python
    cd data-storage/delta-lake/write-to-delta-lake && just static-type-check-python
    cd data-storage/hm-duckdb/query-amazon-s3 && just static-type-check-python
    cd data-storage/hm-duckdb/query-duckdb && just static-type-check-python
    cd data-storage/hm-duckdb/query-lance && just static-type-check-python
    cd data-storage/hm-duckdb/query-parquet && just static-type-check-python
    cd data-storage/hm-duckdb/query-protobuf && just static-type-check-python
    cd data-storage/hm-hdf5 && just static-type-check-python
    cd data-storage/hm-lancedb && just static-type-check-python
    cd data-storage/hm-protobuf && just static-type-check-python
    cd data-storage/lance && just static-type-check-python
    cd data-storage/turso && just static-type-check-python
    cd data-transformation/dbt/projects/dbt_hm_postgres && just static-type-check-python
    cd data-transport/arrow-flight/arrow-flight-client && just static-type-check-python
    cd data-transport/rti-connext-dds && just static-type-check-python
    cd data-visualization/grafana/hm-dashboard && just static-type-check-python
    cd data-visualization/hm-dash/csv-visualizer && just static-type-check-python
    cd data-visualization/hm-dash/parquet-visualizer && just static-type-check-python
    cd data-visualization/hm-plotly/heatmap && just static-type-check-python
    cd data-visualization/hm-pygwalker && just static-type-check-python
    cd data-visualization/iads/iads-data-manager/iads-config-reader && just static-type-check-python
    cd data-visualization/iads/iads-data-manager/iads-data-reader && just static-type-check-python
    cd embedded-system/decode-can-blf-data && just static-type-check-python
    cd embedded-system/decode-can-mf4-data && just static-type-check-python
    cd embedded-system/decode-can-trc-data && just static-type-check-python
    cd embedded-system/format-can-data && just static-type-check-python
    cd embedded-system/hm-serial && just static-type-check-python
    cd embedded-system/pack-unpack-data && just static-type-check-python
    cd embedding/hm-imagebind && just static-type-check-python
    cd embedding/hm-model2vec && just static-type-check-python
    cd embedding/open-clip && just static-type-check-python
    cd ensemble-learning/boosting/hm-catboost && just static-type-check-python
    cd ensemble-learning/boosting/hm-lightgbm && just static-type-check-python
    cd ensemble-learning/boosting/hm-xgboost && just static-type-check-python
    cd evolutionary-algorithm/genetic-algorithm && just static-type-check-python
    cd generative-model/flow-matching && just static-type-check-python
    cd generative-model/neural-ordinary-differential-equation && just static-type-check-python
    cd generative-model/stable-diffusion && just static-type-check-python
    cd generative-model/variational-autoencoder && just static-type-check-python
    cd graph-neural-network && just static-type-check-python
    cd hardware-in-the-loop/national-instruments/hm-pyvisa && just static-type-check-python
    cd hardware-in-the-loop/national-instruments/hm-tdms && just static-type-check-python
    cd hardware-in-the-loop/national-instruments/veristand/hm-veristand && just static-type-check-python
    cd high-performance-computing/hm-jax && just static-type-check-python
    cd infrastructure/ansible && just static-type-check-python
    cd infrastructure/hm-pulumi && just static-type-check-python
    cd infrastructure/opentofu/environments/production/aws/general/files/aws-glue/spark-scripts && just static-type-check-python
    cd large-language-model-application/hm-langgraph/applications/chat-pdf && just static-type-check-python
    cd large-language-model-application/hm-llama-index/applications/chat-pdf && just static-type-check-python
    cd large-language-model-application/hm-pydantic-ai/applications/chat-pdf && just static-type-check-python
    cd large-language-model-architecture/dense-transformer && just static-type-check-python
    cd large-language-model-architecture/mixture-of-experts && just static-type-check-python
    cd large-language-model-evaluation/hm-deepeval && just static-type-check-python
    cd large-language-model-inference/hm-mlx-lm && just static-type-check-python
    cd large-language-model-inference/hm-optimum && just static-type-check-python
    cd large-language-model-inference/hm-sglang/hm-sglang && just static-type-check-python
    cd large-language-model-inference/hm-transformers && just static-type-check-python
    cd large-language-model-inference/kv-caching && just static-type-check-python
    cd large-language-model-inference/speculative-decoding && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/direct-preference-optimization/low-rank-adaptation && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/fine-tune-whisper && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/group-relative-policy-optimization/low-rank-adaptation && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/hm-axolotl && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/hm-llama-factory && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/hm-swift && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/hm-tinker && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/hm-torchtune && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/hm-unsloth && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/proximal-policy-optimization/low-rank-adaptation && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/supervised-fine-tuning/low-rank-adaptation && just static-type-check-python
    cd large-language-model-post-training/fine-tuning/supervised-fine-tuning/quantized-low-rank-adaptation && just static-type-check-python
    cd large-language-model-post-training/post-training-quantization && just static-type-check-python
    cd large-language-model-pre-training/training-optimization/automatic-mixed-precision && just static-type-check-python
    cd large-language-model-pre-training/training-optimization/hm-deepspeed && just static-type-check-python
    cd large-language-model-pre-training/training-paradigm/causal-language-model/qwen3 && just static-type-check-python
    cd large-language-model-pre-training/training-paradigm/masked-language-model/convert-modern-bert-to-core-ml && just static-type-check-python
    cd large-language-model-pre-training/training-paradigm/masked-language-model/modern-bert && just static-type-check-python
    cd large-language-model-pre-training/training-paradigm/masked-language-model/neo-bert && just static-type-check-python
    cd liquid-neural-network && just static-type-check-python
    cd load-testing/hm-locust && just static-type-check-python
    cd machine-learning/configuration-management/hm-hydra-zen && just static-type-check-python
    cd machine-learning/core-ml-tools && just static-type-check-python
    cd machine-learning/feature-store && just static-type-check-python
    cd machine-learning/hm-autogluon && just static-type-check-python
    cd machine-learning/hm-cuml && just static-type-check-python
    cd machine-learning/hm-flax && just static-type-check-python
    cd machine-learning/hm-gradio/applications/classify-image && just static-type-check-python
    cd machine-learning/hm-kubeflow/pipelines/calculate && just static-type-check-python
    cd machine-learning/hm-kubeflow/pipelines/classify-mnist && just static-type-check-python
    cd machine-learning/hm-metaflow/flows/greet && just static-type-check-python
    cd machine-learning/hm-mlflow/experiments/classify-mnist && just static-type-check-python
    cd machine-learning/hm-mlflow/experiments/predict-diabetes && just static-type-check-python
    cd machine-learning/hm-nvidia-modulus && just static-type-check-python
    cd machine-learning/hm-scikit-learn && just static-type-check-python
    cd machine-learning/hm-streamlit/applications/live-line-chart && just static-type-check-python
    cd machine-learning/hm-streamlit/applications/map && just static-type-check-python
    cd machine-learning/hyperparameter-optimization/hm-optuna && just static-type-check-python
    cd machine-learning/model-context-protocol && just static-type-check-python
    cd machine-learning/neural-forecasting/forecast-air-passenger-number && just static-type-check-python
    cd machine-learning/nvidia-dali && just static-type-check-python
    cd machine-learning/nvidia-triton-inference-server/amazon-sagemaker-triton-resnet-50/deploy && just static-type-check-python
    cd machine-learning/nvidia-triton-inference-server/amazon-sagemaker-triton-resnet-50/infer && just static-type-check-python
    cd matlab/call-matlab-function-in-python && just static-type-check-python
    cd multimodal-model/vision-language-action-model/alpamayo-1 && just static-type-check-python
    cd multimodal-model/vision-language-action-model/hm-openpi && just static-type-check-python
    cd multimodal-model/vision-language-action-model/openvla && just static-type-check-python
    cd multimodal-model/vision-language-action-model/vision-language-action/infer && just static-type-check-python
    cd multimodal-model/vision-language-action-model/vision-language-action/train && just static-type-check-python
    cd named-entity-recognition/bert && just static-type-check-python
    cd named-entity-recognition/hm-gliner && just static-type-check-python
    cd parallel-computing/hm-taichi/count-primes && just static-type-check-python
    cd parallel-computing/hm-taichi/fluid-solver && just static-type-check-python
    cd parallel-computing/hm-triton && just static-type-check-python
    cd physics/hm-genesis && just static-type-check-python
    cd quantum-computing && just static-type-check-python
    cd reinforcement-learning/actor-critic-algorithm && just static-type-check-python
    cd reinforcement-learning/q-learning && just static-type-check-python
    cd scientific-computing/hm-sunpy && just static-type-check-python
    cd scientific-computing/surrogate-model && just static-type-check-python
    cd security/hm-prowler && just static-type-check-python
    cd small-language-model/gemma && just static-type-check-python
    cd state-space-model/mamba2 && just static-type-check-python
    cd system-tool/hm-xxhash && just static-type-check-python
    cd tokenization/byte-pair-encoding && just static-type-check-python
    cd vision-language-model/qwen-vl && just static-type-check-python
    cd world-foundation-model/nvidia-cosmos && just static-type-check-python

static-type-check-typescript:
    cd api/api-node && just static-type-check-typescript
    cd ethereum && just static-type-check-typescript
    cd data-visualization/grafana/hm-panel-plugin && just static-type-check-typescript
    cd mobile/mobile-react-native && just static-type-check-typescript
    cd web && just static-type-check-typescript
    cd web-cypress && just static-type-check-typescript
