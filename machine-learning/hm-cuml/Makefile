uv-install-python::
	uv python install
uv-update-lock-file:
	uv lock
uv-install-dependencies:
	uv sync --dev

uv-run-dev:
	export LD_LIBRARY_PATH=/usr/lib/wsl/lib && \
	uv run poe dev
uv-run-test:
	uv run poe test
uv-run-test-coverage:
	uv run poe test-coverage
