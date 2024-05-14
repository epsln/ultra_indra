PYTHON_VERSION ?= "3.11"
POETRY_VERSION ?= "1.7.0"

.PHONY: init
init:
	@poetry env use python${PYTHON_VERSION}
	@poetry install --with dev --with package
	@poetry run pip check
	@poetry run pre-commit install

.PHONY: install
install:
	@poetry install --with dev --with package
	@poetry run pip check

.PHONY: update
update:
	@poetry update --with dev --with package
	@poetry run pip check

.PHONY: run
run:
	@poetry run python main.py 

.PHONY: format
format:
	@poetry run black src/

.PHONY: lint
python-lint:
	@poetry run ruff check --fix --force-exclude src/

.PHONY: type-check
type-check:
	@poetry run mypy --ignore-missing-imports --scripts-are-modules src

.PHONY: quality
quality: format lint type-check
	@:

.PHONY: unit-test
unit-test:
	@poetry run pytest --cov=src tests/unit

.PHONY: integration-test
integration-test:
	@poetry run pytest --cov=src tests/integration

.PHONY: test
test:
	@poetry run pytest --cov=src

.PHONY: build
build:
	@poetry build -f wheel
