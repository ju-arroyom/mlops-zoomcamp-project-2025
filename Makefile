DOCKER_APP_IMAGE = heart_app
DOCKER_MLFLOW_IMAGE = mlflow_server
DOCKER_DB_IMAGE = postgres
DOCKER_TAG ?= latest


# Local poetry install after cloning repo
install_poetry:
	@command -v poetry >/dev/null 2>&1 || { \
		echo "Poetry not found. Installing via pipx..."; \
		pip install --user pipx; \
		pipx install poetry; \
	}
# Configure poetry and create local venv
poetry_env:
	poetry config virtualenvs.in-project true

# Main setup: install poetry, set up venv, install dependencies
setup_poetry_local: install_poetry poetry_env
	poetry install

setup_poetry:
	@echo "Setting up Poetry"
	pipx ensurepath
	pipx install poetry --python $(shell which python)


install_dependecies_no_dev:
	@echo "Installing dependencies"
	poetry lock
	poetry install --no-interaction --no-ansi --without dev

build_app_image:
	docker buildx build -t ${DOCKER_APP_IMAGE}:${DOCKER_TAG} -f docker/Dockerfile-app .

build_mlserver_image:
	docker buildx build -t ${DOCKER_MLFLOW_IMAGE}:${DOCKER_TAG} -f docker/Dockerfile-mlflow .

build_db_local:
	docker buildx build -t ${DOCKER_DB_IMAGE}:${DOCKER_TAG} -f docker/Dockerfile-db-local .

run_db_local: build_db_local
	docker run --name db \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=prediction_metrics \
  -p 5432:5432 \
  -d postgres

up_build:
	docker-compose -f docker/docker-compose.yaml up

down_build:
	docker-compose -f docker/docker-compose.yaml down --volumes --remove-orphans

score_predictions:
	 poetry run python src/mlops/inference/predict.py

# Quality checks

quality_checks:
	isort .
	black .
	ruff check src/mlops

run_unit_tests:
	poetry run pytest -v
