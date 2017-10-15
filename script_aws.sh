#!/bin/sh

echo "Installing Software..."
sleep 1
sudo apt-get update
yes Y | sudo apt-get install gcc make nvidia-cuda-toolkit emacs libmpfr-dev htop
echo "Configuring the GPU..."
sleep 1
sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=0
sudo nvidia-smi -ac 2505,875
#uncomment line below and comment line above
#if decide to run this script on amazon g.3 instance
#sudo nvidia-smi -ac 2505,1177
echo "Now copy the source code and compile"

