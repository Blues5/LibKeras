#!/bin/sh
# Remember to run this script with source, i.e., source ./install_deep_learning.sh

# Environment name
ENV_NAME=libkeras

# Python version (2.7 or 3.5, so they can run Keras)
PYTHON_VER=3.5

# Tensorflow for Python2.7
#TF_PYTHON_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl

# Tensorflow for Python3.5:
TF_PYTHON_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-linux_x86_64.whl

echo "Unpacking and installing Anaconda package:"

# Downloading Anaconda package
wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
# Installing Anaconda4.3 with support to both Python2 and Python3
bash Anaconda3-4.3.1-Linux-x86_64.sh
# Realoding the bash profile, so it will be able to run conda commands without restarting the terminal
source ~/.bashrc

echo "Creating a Anaconda environment:"

# Creating the desired environment with previous specified variables
conda create -n $ENV_NAME python=$PYTHON_VER

echo "Activating the created environment:"

# Activating the recently created environment
source activate $ENV_NAME

echo "Installing tensorflow:"

# Installing tensorflow package according to current environment python's version
pip install --ignore-installed --upgrade $TF_PYTHON_URL

echo "Installing keras:"

# Installing latest keras package
pip install keras

echo "Upgrading theano to work with cuDNN 5.1:"

# Uninstalling previous theano, as it will not work with cuDNN 5.1
pip uninstall theano
# Fetching theano development version from github, as it will work with cuDNN 5.1
pip install --no-deps git+https://github.com/Theano/Theano.git#egg=Theano

echo "Installing additional packages for LibKeras:"

# Installing additional packages for LibKeras
pip install h5py
pip install pillow
pip install pandas
pip install matplotlib
pip install sklearn

echo "Deactivating the current environment:"

# Deactivating the current environment
source deactivate $ENV_NAME
