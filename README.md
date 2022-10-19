# pytorch-gpu-data-science-project

Repository containing scaffolding for a Python 3-based data science project with GPU acceleration using the [PyTorch](https://pytorch.org/) ecosystem. 

## Creating a new project from this template

Simply follow the [instructions](https://help.github.com/en/articles/creating-a-repository-from-a-template) to create a new project repository from this template.

## Project organization

Project organization is based on ideas from [_Good Enough Practices for Scientific Computing_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510).

1. Put each project in its own directory, which is named after the project.
2. Put external scripts or compiled programs in the `bin` directory.
3. Put raw data and metadata in a `data` directory.
4. Put text documents associated with the project in the `doc` directory.
5. Put all Docker related files in the `docker` directory.
6. Install the Conda environment into an `env` directory. 
7. Put all notebooks in the `notebooks` directory.
8. Put files generated during cleanup and analysis in a `results` directory.
9. Put project source code in the `src` directory.
10. Name all files to reflect their content or function.

## Building the Conda environment

After adding any necessary dependencies that should be downloaded via `conda` to the 
`environment.yml` file and any dependencies that should be downloaded via `pip` to the 
`requirements.txt` file you create the Conda environment in a sub-directory `./env`of your project 
directory by running the following commands.

```bash
export ENV_PREFIX=$PWD/env
mamba env create --prefix $ENV_PREFIX --file environment.yml --force
```

Once the new environment has been created you can activate the environment with the following 
command.

```bash
conda activate $ENV_PREFIX
```

Note that the `ENV_PREFIX` directory is *not* under version control as it can always be re-created as 
necessary.

For your convenience these commands have been combined in a shell script `./bin/create-conda-env.sh`. 
Running the shell script will create the Conda environment, activate the Conda environment, and build 
JupyterLab with any additional extensions. The script should be run from the project root directory 
as follows. 

```bash
./bin/create-conda-env.sh
```

### Ibex

The most efficient way to build Conda environments on Ibex is to launch the environment creation script 
as a job on the debug partition via Slurm. For your convenience a Slurm job script 
`./bin/create-conda-env.sbatch` is included. The script should be run from the project root directory 
as follows.

```bash
sbatch ./bin/create-conda-env.sbatch
```

### Listing the full contents of the Conda environment

The list of explicit dependencies for the project are listed in the `environment.yml` file. To see 
the full lost of packages installed into the environment run the following command.

```bash
conda list --prefix $ENV_PREFIX
```

### Updating the Conda environment

If you add (remove) dependencies to (from) the `environment.yml` file or the `requirements.txt` file 
after the environment has already been created, then you can re-create the environment with the 
following command.

```bash
$ mamba env create --prefix $ENV_PREFIX --file environment.yml --force
```

## Installing the NVIDIA CUDA Compiler (NVCC) (Optional)

Installing the NVIDIA CUDA Toolkit manually is only required if your project needs to use the `nvcc` compiler. 
Note that even if you have not written any custom CUDA code that needs to be compiled with `nvcc`, if your project 
uses packages that include custom CUDA extensions for PyTorch then you will need `nvcc` installed in order to build these packages.

If you don't need `nvcc`, then you can skip this section as `conda` will install a `cudatoolkit` package 
which includes all the necessary runtime CUDA dependencies (but not the `nvcc` compiler).

### Workstation

You will need to have the [appropriate version](https://developer.nvidia.com/cuda-toolkit-archive) 
of the NVIDIA CUDA Toolkit installed on your workstation. If using the most recent versionf of PyTorch, then you 
should install [NVIDIA CUDA Toolkit 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) 
[(documentation)](https://docs.nvidia.com/cuda/archive/11.1.1/).

After installing the appropriate version of the NVIDIA CUDA Toolkit you will need to set the 
following environment variables.

```bash
$ export CUDA_HOME=/usr/local/cuda-11.1
$ export PATH=$CUDA_HOME/bin:$PATH
$ export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Ibex

Ibex users do not neet to install NVIDIA CUDA Toolkit as the relevant versions have already been 
made available on Ibex by the Ibex Systems team. Users simply need to load the appropriate version 
using the `module` tool. 

```bash
$ module load cuda/11.1.1
```

## Using Docker

In order to build Docker images for your project and run containers with GPU acceleration you will 
need to install 
[Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/), 
[Docker Compose](https://docs.docker.com/compose/install/) and the 
[NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker).

Detailed instructions for using Docker to build and image and launch containers can be found in 
the `docker/README.md`.
