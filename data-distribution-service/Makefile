poetry-env-use:
	poetry env use python3.10
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-build:
	poetry run poe build
poetry-run-subscribe:
	poetry run poe subscribe
poetry-run-publish:
	poetry run poe publish
poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

rti-convert:
	rtiddsgen -convertToXml hm_message.idl -d .
