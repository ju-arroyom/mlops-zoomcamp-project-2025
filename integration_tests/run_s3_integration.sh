#!/usr/bin/env bash

if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"
fi

# Important vars
BUCKET=s3://heart-app
NUM_TRIALS=5
DOCKER_MLFLOW_IMAGE=mlflow_server
DOCKER_TAG=latest
# Set environmental vars
export S3_BUCKET=heart-app
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export MLFLOW_TRACKING_URI=http://localhost:5500
export OPTUNA_EXPERIMENT=xgb_optuna_search
export PREFECT_API_URL=http://localhost:4200/api # Crucial for connection
export STORAGE_TYPE=s3

# Build mlflow server image
docker buildx build -t ${DOCKER_MLFLOW_IMAGE}:${DOCKER_TAG} -f ../docker/Dockerfile-mlflow ..

docker-compose \
  -f ../docker/docker-compose.yaml \
  up -d mlflow prefect-ui

docker-compose \
 -f docker-compose.localstack.yml up -d \
#docker-compose -f docker-compose.localstack.yml up -d

# Create bucket
echo "Creating Bucket: ${BUCKET}"
aws --endpoint-url=http://localhost:4566 s3 mb $BUCKET

# Wait a little bit
sleep 1

# Copy data to bucket

aws --endpoint-url=http://localhost:4566 s3 cp ../src/mlops/data/cardiac_arrest_dataset.csv $BUCKET/data/cardiac_arrest_dataset.csv

# Check size of input file
echo "Reviewing file was added"
aws --endpoint-url=http://localhost:4566 s3 ls $BUCKET --recursive

# Run trainer
echo "Running optuna 🧪 for: $NUM_TRIALS trials"
poetry run python ../src/mlops/pipelines/training_pipeline.py --num_trials $NUM_TRIALS

# Check size of input file
FILES=$(aws --endpoint-url=http://localhost:4566 s3 ls $BUCKET/data/ --recursive)

# Count number of files
FILE_COUNT=$(echo "$FILES" | wc -l)

# Output results
if [ "$FILE_COUNT" -eq 3 ]; then
    echo "✅ 3 files found:"
    echo "$FILES" | awk '{print $4}'  # prints just the filenames
else
    echo "❌ Unexpected number of files: $FILE_COUNT"
    echo "Files found:"
    echo "$FILES" | awk '{print $4}'
    exit 1
fi
# Wait a little bit
sleep 1

# Bring down containers
docker-compose \
  -f ../docker/docker-compose.yaml down &&
  docker-compose -f docker-compose.localstack.yml down
