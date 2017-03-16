#!/bin/sh
# Remember to run this script with sudo, i.e., sudo ./install_cuda.sh

echo "Unpacking and installing CUDA Toolkit package:"

# Unpacking CUDA package and adding to apt-get repository
dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64.deb
# Updating apt-get repository
apt-get update
# Installing CUDA Toolkit package
apt-get install cuda

echo "Unpacking and installing cuDNN package:"

# Extracting cuDNN package
tar -zxf cudnn-8.0-linux-x64-v5.1.tgz
# Entering its folder
cd cuda
# Copying all the needed files to CUDA's main folders
cp lib64/* /usr/local/cuda/lib64/
cp include/* /usr/local/cuda/include/

echo "Installing CUDA Profile Tools package:"

# Installing CUDA Profile Tools package, also needed to run the GPU
apt-get install libcupti-dev

echo "Adding environment variables:"

# Adding CUDA environment variables to bash profile
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc
# Adding CUDA environment variable to work with tensorflow
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
# Adding CUDA environment variable to work with theano
echo 'export CUDA_ROOT=/usr/local/cuda' >> ~/.bashrc
# Realoding the bash profile, so it will be able to load the variables without restarting the terminal
source ~/.bashrc

echo "Testing GPU:"

# Entering CUDA's utilities folder
cd ~/../../usr/local/cuda/samples/1_Utilities/deviceQuery
# Making all .c files
make
# Running deviceQuery script to check whether the GPU is working or not
./deviceQuery
# Cleaning all compiled files
make clean
