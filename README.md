[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KAUST-Academy/introduction-to-machine-learning/HEAD)

# Introduction to Deep Learning

There is strong demand for deep learning skills and expertise to solve challenging business problems both globally and locally in KSA. This course will help learners build capacity in core DL tools and methods and enable them to develop their own deep learning applications. This course covers the basic theory behind DL algorithms but the majority of the focus is on hands-on examples using [PyTorch](https://pytorch.org/).

## Learning Objectives

The primary learning objective of this course is to provide students with practical, hands-on experience with state-of-the-art machine learning and deep learning tools that are widely used in industry.

This course covers portions of chapters 10-19 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) and chapters 11-19 of [Machine Learning with PyTorch and Scikit-Learn](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312). The following topics will be discussed.

* Introduction to Artificial Neural Networks (ANNs) 
* Training Deep Neural Networks (DNNs) 
* Custom Models and Training with PyTorch and Lightning 
* Stratgeies for Loading and Preprocessing Data
* Training and Deploying PyTorch Models at Scale 

## Lectures

### Lecture 1: An Introduction to Artificial Neural Networks [(Slides)](https://kaust-my.sharepoint.com/:p:/g/personal/pughdr_kaust_edu_sa/EfRHWqkIFjpBk9j8aL4I3fABUngt5d3uccvxjDuuurtYfA?e=ffNqUj)

In this lecture we cover the fundamental concepts behind artificial neural networks (ANNs), beginning with the biological inspiration and simple perceptrons, then moving to multi-layer perceptrons (MLPs) and the back-propagation algorithm for training them. The hands on labs will cover the fundamental concepts of PyTorch and then demonstrate how to implement foundational models for supervised learning problems such as linear and logistic regression, and MLPs from scratch with PyTorch as well as how to train these models using full-batch, gradient descent.

**Suggested reading:**

* Chapter 9 of [Hands-on Machine Learning with Scikit-Learn and PyTorch](https://www.oreilly.com/library/view/hands-on-machine-learning/9798341607972/) 
* Chapter 11 of [ML with PyTorch and Sklearn](https://www.oreilly.com/library/view/machine-learning-with/9781801819312/)

| **Tutorial** | **Open in Google Colab** | **Open in Kaggle** |
|--------------|:------------------------:|:------------------:|
| First Steps with PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01a-first-steps-with-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01a-first-steps-with-pytorch.ipynb) |
| Linear Regression from Scratch with PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01b-linear-regression-from-scratch-with-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01b-linear-regression-from-scratch-with-pytorch.ipynb) |
| Logistic Regression from Scratch with PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01c-logistic-regression-from-scratch-with-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01c-logistic-regression-from-scratch-with-pytorch.ipynb) |
| MLPs for Regression with PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01d-mlp-for-regression-with-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01d-mlp-for-regression-with-pytorch.ipynb) | 
| MLPs for Classification with PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01e-mlp-for-classification-with-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01e-mlp-for-classification-with-pytorch.ipynb) | 

### Lecture 2: Building Neural Networks with PyTorch

In these hands on labs we will discusses how to build and train multi-layter perceptron (MLP) models using the PyTorch API—choosing number of layers and neurons, activation functions, loss functions, and optimizers—and illustrates how these networks form the basis of deep learning. Additional labs cover building data pipelines with PyTorch and model training using mini-batch stochastic gradient descent. This lecture emphasizes practical implementation in PyTorch and helps the student move from shallow models to basic deep architectures, preparing the ground for more advanced topics.

| **Tutorial** | **Open in Google Colab** | **Open in Kaggle** |
|--------------|:------------------------:|:------------------:|
| Building Data Pipelines with PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/02a-building-data-pipelines-in-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/02a-building-data-pipelines-in-pytorch.ipynb) | 
| Mini-batch Gradient Descent with PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/02b-implementing-minibatch-gradient-descent.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/notebooks/02b-implementing-minibatch-gradient-descent.ipynb) | 
| Model Evaluation with PyTorch and Torchmetrics | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/02c-model-evaluation.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/notebooks/02c-model-evaluation.ipynb) | 
| Building an Image Classifier in PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/02d-building-an-image-classifier-with-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/notebooks/02d-building-an-image-classifier-with-pytorch.ipynb) | 
| Saving and Loading PyTorch Models | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/02e-saving-and-loading-pytorch-models.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/notebooks/02e-saving-and-loading-pytorch-models.ipynb) |

### Lecture 3: Training Neural Networks, Part I: Stochastic Gradient Descent [(Slides)](https://kaust-my.sharepoint.com/:p:/g/personal/pughdr_kaust_edu_sa/ESk6HlHXmytAipnVVIeTCtsBnEVZLQeF4KNpE7A49GZCcA?e=VteHn9)

TBD

**Suggested reading:**

* Chapter 4 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) 
* Chapter ? of [ML with PyTorch and Sklearn](https://www.oreilly.com/library/view/machine-learning-with/9781801819312/)

| **Tutorial** | **Open in Google Colab** | **Open in Kaggle** |
|--------------|:------------------------:|:------------------:|
| Introduction to PyTorch Lightning | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01e-introduction-to-pytorch-lightning.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/davidrpugh/introduction-to-deep-learning/blob/master/notebooks/01e-introduction-to-pytorch-lightning.ipynb) | 
| Introduction to PyTorch Lightning | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/01e-introduction-to-pytorch-lightning.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/01e-introduction-to-pytorch-lightning.ipynb)


### Lecture 4: Training Neural Networks, Part II [(Slides)](https://kaust-my.sharepoint.com/:p:/g/personal/pughdr_kaust_edu_sa/EeA0M9ydAEVGmfAWZhzVDEwBWtnOVos3YroubDAkfswWoQ?e=ryFLJy)

* Consolidation of previous days content via Q/A and live coding demonstrations.
* The morning session will focus on the theory behind neural networks for solving both classification and regression problems by covering chapters 12-13 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) and chapters 12-13 of [Machine Learning with PyTorch and Scikit-Learn](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312).  
* The afternoon session will focus on applying the techniques learned in the morning session using [PyTorch](https://pytorch.org/), followed by a short assessment on the Kaggle data science competition platform.

| **Tutorial** | **Open in Google Colab** | **Open in Kaggle** |
|--------------|:------------------------:|:------------------:|
| Deep Dive into PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/01d-deep-dive-into-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/01d-deep-dive-into-pytorch.ipynb) | 
| Vanishing and Exploding Gradients | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02c-vanishing-exploding-gradients.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02c-vanishing-exploding-gradients.ipynb) | 
| Transfer Learning and Unsupervised Pre-training | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02d-transfer-learning-and-unsupervised-pretraining.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02d-transfer-learning-and-unsupervised-pretraining.ipynb) | 
| Faster Optimizers | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02e-faster-optimizers.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02e-faster-optimizers.ipynb) | 
| Learning Rate Schedulers | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02f-learning-rate-schedulers.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02f-learning-rate-schedulers.ipynb) | 
| Regularization | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02g-regularization.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/02g-regularization.ipynb) | 

### Module 3: [Deploying and Scaling PyTorch Models](https://kaust-my.sharepoint.com/:p:/g/personal/pughdr_kaust_edu_sa/EZItqTR1l3VOs0UzJJ7NAVABaDT2FN5kclARg2Rv8cnoxA?e=vyaLcu)

* Consolidation of previous days content via Q/A and live coding demonstrations.  
* The morning session will focus on various topics related to training and deploying PyTorch models as scale by covering chapter 19 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). 
* The afternoon session will allow time for a final assessment as well as additional time for learners to complete any of the previous assessments.

## Assessment

Student performance on the course will be assessed through participation in a Kaggle classroom competition. 

# Repository Organization

Repository organization is based on ideas from [_Good Enough Practices for Scientific Computing_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510).

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

## Using Docker

In order to build Docker images for your project and run containers with GPU acceleration you will 
need to install 
[Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/), 
[Docker Compose](https://docs.docker.com/compose/install/) and the 
[NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker).

Detailed instructions for using Docker to build and image and launch containers can be found in 
the `docker/README.md`.
