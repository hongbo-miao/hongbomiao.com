uv-install-python:
	uv python install
uv-update-lock-file:
	uv lock
uv-install-dependencies:
	uv sync --dev

uv-run-dev:
	export PYOPENGL_PLATFORM=glx && \
	uv run poe dev
uv-run-test:
	uv run poe test
uv-run-test-coverage:
	uv run poe test-coverage
