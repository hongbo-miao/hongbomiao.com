cargo-clean:
	cargo clean
cargo-update:
	cargo update
cargo-check:
	cargo check
cargo-test:
	cargo test --all-features
cargo-build-development:
	cargo build
cargo-build-production:
	cargo build --release
cargo-run-development:
	cargo run
cargo-run-production:
	cargo run --release

lint-rust-rustfmt:
	cargo fmt --all -- --check
lint-rust-rustfmt-fix:
	cargo fmt --all
lint-rust-clippy:
	cargo clippy
lint-rust-clippy-fix:
	cargo clippy --fix --allow-dirty --allow-staged

docker-build:
	cd .. && \
	docker build --file=api-rust/Dockerfile --tag=ghcr.io/hongbo-miao/hm-api-rust:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-api-rust:latest
docker-run:
	docker run --publish=36147:36147 --name=hm-api-rust --rm ghcr.io/hongbo-miao/hm-api-rust:latest

brew-install-opencv::
	brew install opencv

uv-install-python:
	uv python install
uv-update-lock-file:
	uv lock
uv-install-dependencies:
	uv sync --dev
uv-run-download-model:
	uv run poe download-model
