#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# path to the Conda environment
ENV_PREFIX="$PROJECT_DIR"/env

# project should have a data directory
DATA_DIR="$PROJECT_DIR"/data

# creates a separate directory for each job
JOB_NAME=example-training-job
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# launch the training job
CPUS_PER_GPU=6
sbatch --job-name "$JOB_NAME" --cpus-per-gpu $CPUS_PER_GPU \
    "$PROJECT_DIR"/bin/train.sbatch "$ENV_PREFIX" \
        "$PROJECT_DIR"/src/train-argparse.py \
            --dataloader-num-workers $CPUS_PER_GPU \
            --data-dir "$DATA_DIR" \
            --num-training-epochs 10 \
            --output-dir "$JOB_RESULTS_DIR" \
            --tqdm-disable
