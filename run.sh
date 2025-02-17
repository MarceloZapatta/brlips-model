#!/bin/bash

# Build the Docker image
docker-compose build

# Function to run training
train() {
    docker-compose run --rm lipreader python train.py
}

# Function to run testing
test() {
    docker-compose run --rm lipreader python test.py
}

# Parse command line arguments
case "$1" in
    "train")
        train
        ;;
    "test")
        test
        ;;
    *)
        echo "Usage: ./run.sh [train|test]"
        exit 1
        ;;
esac 