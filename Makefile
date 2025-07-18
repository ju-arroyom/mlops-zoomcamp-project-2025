DOCKER_APP_IMAGE = heart_app
DOCKER_MLFLOW_IMAGE = mlflow_server
DOCKER_DB_IMAGE = postgres
DOCKER_TAG ?= latest


setup_poetry:
	@echo "Setting up Poetry"
	@if [ "$(ROOT)" = "true" ]; then \
		apt-get update; \
		apt-get install -y pipx; \
	else \
		sudo apt-get update; \
		sudo apt-get install -y pipx; \
	fi
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

up_build: build_app_image build_mlserver_image
	docker-compose -f docker/docker-compose.yaml up

down_build:
	docker-compose -f docker/docker-compose.yaml down

score_predictions:
	 poetry run python src/mlops/inference/predict.py

# Local test section

mlflow_server:
	poetry run mlflow server --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5500

fast_api_server:
	poetry run python app/app.py

prefect_server:
	poetry run prefect server start --host 0.0.0.0 --port 4200

run_training_pipeline:
	MLFLOW_TRACKING_URI=http://localhost:5500 \
	OPTUNA_EXPERIMENT=xgb_optuna_search \
	poetry run python src/mlops/pipelines/training_pipeline.py

dashboard: run_db_local score_predictions
	poetry run streamlit run src/mlops/monitoring/dashboard.py --server.port=8501 