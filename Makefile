.PHONY: clean clean-test clean-pyc clean-build docs help airflow-init airflow-test airflow-list airflow-run airflow-clean
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
	uv venv --python 3.12
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

# =============================================================================
# Airflow Commands
# =============================================================================

AIRFLOW_HOME ?= $(shell pwd)/src/typewiki/airflow

airflow-init: ## Initialize Airflow database (run once after setup)
	@echo ----------------------------------------------------------------
	@echo INITIALIZING AIRFLOW DATABASE...
	AIRFLOW_HOME=$(AIRFLOW_HOME) uv run airflow db migrate
	@echo AIRFLOW DATABASE INITIALIZED
	@echo ----------------------------------------------------------------

airflow-test: ## Test the Help Center ingest DAG
	@echo ----------------------------------------------------------------
	@echo TESTING HELP CENTER INGEST DAG...
	AIRFLOW_HOME=$(AIRFLOW_HOME) uv run airflow dags test typewiki_helpcenter_ingest $$(date +%Y-%m-%d)
	@echo DAG TEST COMPLETE
	@echo ----------------------------------------------------------------

airflow-list: ## List all available DAGs
	AIRFLOW_HOME=$(AIRFLOW_HOME) uv run airflow dags list

airflow-run: ## Start Airflow standalone (webserver + scheduler)
	@echo ----------------------------------------------------------------
	@echo STARTING AIRFLOW STANDALONE...
	@echo Access the UI at http://localhost:8080
	@echo ----------------------------------------------------------------
	AIRFLOW_HOME=$(AIRFLOW_HOME) uv run airflow standalone

airflow-clean: ## Remove Airflow artifacts (database, logs)
	@echo ----------------------------------------------------------------
	@echo CLEANING AIRFLOW ARTIFACTS...
	rm -f $(AIRFLOW_HOME)/airflow.db
	rm -rf $(AIRFLOW_HOME)/logs/
	rm -f $(AIRFLOW_HOME)/airflow-webserver.pid
	rm -f $(AIRFLOW_HOME)/standalone_admin_password.txt
	@echo AIRFLOW ARTIFACTS CLEANED
	@echo ----------------------------------------------------------------