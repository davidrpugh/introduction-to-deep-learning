#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# path to Conda environment
ENV_PREFIX="$PROJECT_DIR"/env

# data should be read from a data directory
DATA_DIR="$PROJECT_DIR"/data

# creates a separate directory for each job
JOB_NAME=example-training-job
JOB_RESULTS_DIR="$PROJECT_DIR"/results/"$JOB_NAME"
mkdir -p "$JOB_RESULTS_DIR"

# create a directory to store the checkpoints
CHECKPOINTS_DIR="$JOB_RESULTS_DIR"/checkpoints
mkdir -p "$CHECKPOINTS_DIR"

# use a single file to track intermediate checkpoints
CHECKPOINT_FILEPATH="$CHECKPOINTS_DIR"/checkpoint.pt

# define number of training periods and training epochs (per period)
NUM_TRAINING_PERIODS=10
NUM_EPOCHS_PER_PERIOD=1

# launch the training job for the initial period
CPUS_PER_GPU=4
TRAIN_JOBID=$(
    sbatch --job-name "$JOB_NAME" --cpus-per-gpu $CPUS_PER_GPU --parsable \
        "$PROJECT_DIR"/bin/train.sbatch "$ENV_PREFIX" \
            "$PROJECT_DIR"/src/train-checkpoint-restart.py \
                --dataloader-num-workers $CPUS_PER_GPU \
                --data-dir "$DATA_DIR" \
                --num-training-epochs $NUM_EPOCHS_PER_PERIOD \
                --tqdm-disable \
                --write-checkpoint-to "$CHECKPOINT_FILEPATH" \
)

# store the most recent checkpoint
cp "$CHECKPOINT_FILEPATH" "$CHECKPOINTS_DIR"/most-recent-checkpoint.pt

# queue the training jobs for the remaining periods
for ((PERIOD=1;PERIOD<$NUM_TRAINING_PERIODS;PERIOD++))
do

    TRAIN_JOBID=$(
        sbatch --job-name "$JOB_NAME" --cpus-per-gpu $CPUS_PER_GPU --parsable --dependency=afterok:$TRAIN_JOBID --kill-on-invalid-dep=yes \
            "$PROJECT_DIR"/bin/train.sbatch "$ENV_PREFIX" \
                "$PROJECT_DIR"/src/train-checkpoint-restart.py \
                    --checkpoint-filepath "$CHECKPOINTS_DIR"/most-recent-checkpoint.pt \
                    --dataloader-num-workers $CPUS_PER_GPU \
                    --data-dir "$DATA_DIR" \
                    --num-training-epochs $NUM_EPOCHS_PER_PERIOD \
                    --tqdm-disable \
                    --write-checkpoint-to "$CHECKPOINT_FILEPATH" \
    )

    # store the most recent checkpoint
    cp "$CHECKPOINT_FILEPATH" "$CHECKPOINTS_DIR"/most-recent-checkpoint.pt

done
