## Creating the Conda environment

For your convenience the commands to create the Conda environment have been combined in a shell script. The script should be run from the project root directory as follows. 

```bash
./bin/create-conda-env.sh
```

## Launching a job via Slurm to create the Conda environment

While running the shell script above on a login node will create the Conda environment, you may prefer to launch a job via Slurm
to create the Conda environment. If you lose your connection to the Ibex login node whilst your Conda environment script is running 
the environment will be left in an inconsistent state and you will need to start over. Depending on the load on the Ibex login nodes, 
lanuching a job via Slurm to create your Conda environment can also be faster.

For your convenience the commands to launch a job via Slurm to create the Conda environment have been combined into a job script. The script should be run from the project root directory as follows. 

```bash
sbatch ./bin/create-conda-env.sbatch
```

## Launching a Jupyter server for interactive work

The job script `launch-jupyter-server.sbatch` launches a Jupyter server for interactive prototyping. To launch a JupyterLab server 
use `sbatch` to submit the job script by running the following command from the project root directory.

```bash
sbatch ./bin/launch-jupyter-server.sbatch
```

If you prefer the classic Jupyter Notebook interface, then you can launch the Jupyter notebook server with the following command in 
the project root directory.

```bash
sbatch ./bin/launch-jupyter-server.sbatch notebook
```

Once the job has started, you can inspect the `./bin/launch-jupyter-server-$SLURM_JOB_ID-slurm.err` file where you will find 
instructions on how to access the server running in your local browser.

### SSH tunneling between your local machine and Ibex compute node(s)
To connect to the compute node on Ibex running your Jupyter server, you need to create an SSH tunnel from your local machine 
to a login node on Ibex using a command similar to the following.

```
ssh -L ${JUPYTER_PORT}:${IBEX_NODE}:${JUPYTER_PORT} ${USER}@glogin.ibex.kaust.edu.sa
```

The exact command for your job can be copied from the `./bin/launch-jupyter-server-$SLURM_JOB_ID-slurm.err` file.

### Accessing the Jupyter server from your local machine

Once you have set up your SSH tunnel, in order to access the Jupyter server from your local machine you need to copy the 
second URL provided in the Jupyter server logs in the `launch-jupyter-server-$SLURM_JOB_ID-slurm.err` file and paste it into 
the browser on your local machine. The URL will look similar to the following.

```
http://127.0.0.1:$JUPYTER_PORT/lab?token=$JUPYTER_TOKEN
```

The exact command for your job containing both your assigned `$JUPYTER_PORT` as well as your specific `$JUPYTER_TOKEN` can 
be copied from the `launch-jupyter-server-$SLURM_JOB_ID-slurm.err`.

## Launching a VS Code server for development work

The job script `launch-code-server.sbatch` launches a Microsoft Visual Studio (VS) Code server for development work. In order to 
use VS Code server, you will first need to install the server package in your Ibex home directory following the instructions 
provided on [GitHub](https://github.com/kaust-rccl/ibex-code-server-install). Once you have installed VS Code server, you can 
launch a server by running the following command from the project root directory.

```bash
sbatch ./bin/launch-code-server.sbatch
```

Once the job has started, you can inspect the `./bin/launch-code-server-$SLURM_JOB_ID-slurm.err` file where you will find 
instructions on how to access the server running in your local browser.

### SSH tunneling between your local machine and Ibex compute node(s)
To connect to the compute node on Ibex running your VS Code server, you need to create an SSH tunnel from your local machine 
to a login node on Ibex using a command similar to the following.

```
ssh -L ${JUPYTER_PORT}:${IBEX_NODE}:${JUPYTER_PORT} ${USER}@glogin.ibex.kaust.edu.sa
```

The exact command for your job can be copied from the `./bin/launch-code-server-$SLURM_JOB_ID-slurm.err` file.

### Accessing the VS Code server from your local machine

Once you have set up your SSH tunnel, in order to access the VS Code server from your local machine you need to copy the 
second URL provided in the `launch-code-server-$SLURM_JOB_ID-slurm.err` file and paste it into the browser on your local 
machine. The URL will look similar to the following.

```
localhost:$CODE_SERVER_PORT
```

The exact command for your job containing both your assigned `$CODE_SERVER_PORT` can be copied from the 
`launch-code-server-$SLURM_JOB_ID-slurm.err`.

## Launching a training job via Slurm

The `src` directory contains an example training script, `train.py`, that trains a classification pipeline on the 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. You can launch this training script as a batch 
job on Ibex via Slurm using the following command in the project root directory.

```bash
./bin/launch-train.sh
```

### The `./bin/launch-train.sh` script

Wrapping your job submission inside a `launch-train.sh` script is an Ibex "best-practice" that will help you automate more 
complex machine learning workflows. In particular, this example script demonstrates how to break up a single training job 
into training periods (where each training period consists of one or more training epochs) and launches a sequence of jobs 
for each training period. Breaking up large training jobs into smaller jobs can dramatically improve your job throughput! 

### The `./bin/train.sbatch` script

The script `./bin/train.sbatch` is the actual Slurm job script. This script can be broken down into several parts that 
are common to all machine learning jobs on Ibex.

#### Request resources from Slurm

You will request resources for your job using Slurm headers. It is important to request a balanced allocation of CPUs, GPUs, 
and CPU memory in order to insure good overall job performance. You should typically request resources that are roughly 
proportional to the amount of GPUs you are requesting. Most of our nodes have 8 V100 GPUs, 48 CPU cores, and 748 GB of CPU 
memory. The headers below request 6 Intel CPU cores per NVIDIA V100 GPU, and 64G of CPU memory for 2 hours.   

```bash
#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=4  
#SBATCH --mem=64G
#SBATCH --partition=batch 
#SBATCH --mail-type=ALL
#SBATCH --output=results/%x/%j-slurm.out
#SBATCH --error=results/%x/%j-slurm.err
```

#### Activate the Conda environment

Activating the Conda environment is done in the usual way however it is critical for the job script to run inside a 
login shell in order for the `conda activate` command to work as expected (this is why the first line of the job script 
is `#!/bin/bash --login`). It is also a good practice to purge any modules that you might have loaded prior to launching 
the training job. Note that this script expects the first argument to be the path to the Conda environment.
 
```bash
module purge
conda activate "$1"
```

#### Starting the NVDashboard monitoring server

After activating the Conda environment, but before launching your training script, you should start the 
NVDashboard monitoring server to run in the background using `srun` to reserve a free port. The 
`./bin/launch-nvdashboard-server.srun` script launches the monitoring server and logs out the assigned 
port to the `slurm.err` file for the job.

```bash
# use srun to launch NVDashboard server in order to reserve a port
srun --resv-ports=1 ./bin/launch-nvdashboard-server.srun &
NVDASHBOARD_PID=$!
```

#### Launch a Python training script

Finally, you launch the training job! Note that we use the special Bash variable `"${@:2}"` to refer to all of the 
command line arguments (other than the first!) passed to the Slurm job script. This allows you to reuse the same Slurm 
job script for other training jobs!

```bash
python "${@:2}"
```

#### Stopping the NVDashboard monitoring server

Once the training script has finished running, you should stop the NVDashboard server so that your job exits. If 
you do not stop the server the job will continue to run until it reached its time limit (which wastes resources).

```
# shutdown the NVDashboard server
kill $NVDASHBOARD_PID
```
