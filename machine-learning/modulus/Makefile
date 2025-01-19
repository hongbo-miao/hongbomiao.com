uv-install-python::
	uv python install
uv-update-lock-file:
	uv lock
uv-install-dependencies:
	uv sync --extra=build
	uv sync --extra=build --extra=compile
	uv sync --extra=build --extra=compile --dev

uv-run-dev:
	uv run poe dev
uv-run-test:
	uv run poe test
uv-run-test-coverage:
	uv run poe test-coverage
