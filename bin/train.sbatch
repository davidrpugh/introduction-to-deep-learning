#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=4  
#SBATCH --mem=64G 
#SBATCH --partition=batch 
#SBATCH --mail-type=ALL
#SBATCH --output=results/%x/%j-slurm.out
#SBATCH --error=results/%x/%j-slurm.err

# entire script fails if single command fails
set -e

# activate the conda environment
module purge
conda activate "$1"

# use srun to launch NVDashboard server in order to reserve a port
srun --resv-ports=1 ./bin/launch-nvdashboard-server.srun &
NVDASHBOARD_PID=$!

# launch the training script
python "${@:2}"

# shutdown the NVDashboard server
kill $NVDASHBOARD_PID
