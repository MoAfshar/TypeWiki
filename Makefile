.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

help:
	@python3 -x "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: ## Remove all build, test, coverage and python artifacts
	@echo ----------------------------------------------------------------
	@echo CLEANING UP ...
	make clean-build clean-pyc clean-test
	@echo ALL CLEAN.
	@echo ----------------------------------------------------------------

clean-build: ## Remove build artifacts
	@echo Cleaning build artifacts ...
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	@echo Clean pyc file artifacts ...
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	@echo cleaning test artifacts ...
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## Check style with flake8
	install-pre-commit
	@echo ----------------------------------------------------------------
	@echo RUNNING FLAKE8 ...
	flake8
	@echo YOUR CODE IS PEP8 COMPLIANT, LETS GOO!!
	@echo ----------------------------------------------------------------

test: ## Run tests (and coverage if configured in setup.cfg) with the default Python
	@echo ----------------------------------------------------------------
	@echo RUNNING TESTS ...
	pytest --cov=typewiki
	@echo TESTS PASS, CONGRATS!
	@echo ----------------------------------------------------------------

coverage: ## Check code coverage quickly with the default Python
	@echo Producing coverage report at COVERAGE.txt...
	coverage report > COVERAGE.txt

dist: clean ## Builds source and wheel package
	uv build
	ls -l dist

venv: ## Create a virtual environment
	@echo ---------------------------------------------------------------
	@echo CREATING VIRTUAL ENVIRONMENT...
	uv venv
	@echo VIRTUAL ENVIRONMENT CREATED
	@echo Run 'source .venv/bin/activate' to activate it
	@echo ---------------------------------------------------------------


install-dev-local: ## Install all the stuff you need to develop locally
	uv sync --all-extras
	pre-commit install
	pre-commit autoupdate
	PIP_PREFER_BINARY=true pre-commit install-hooks


install: clean ## Install the package to the active Python's site-package via pip
	@echo ----------------------------------------------------------------
	@echo INSTALLING TypeWiki ...
	uv sync --no-editable
	@echo INSTALLED TypeWiki
	@echo ----------------------------------------------------------------
	@echo TypeWiki info:
	@echo ----------------------------------------------------------------
	pip show typewiki
	@echo ----------------------------------------------------------------