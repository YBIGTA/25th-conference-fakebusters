#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Install Miniconda (if not already installed)
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    chmod +x ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    ~/miniconda3/bin/conda init bash
fi

# Reload shell for Conda initialization
echo "Reloading shell..."
source ~/miniconda3/etc/profile.d/conda.sh

# Step 2: Set up Conda environment
echo "Setting up Conda environment..."
conda create -n video python==3.11 -y || true
conda activate video

# Step 3: Install dependencies
echo "Installing dependencies..."
conda install conda-forge::libgl -y
conda install -c conda-forge ffmpeg -y
apt update
pip install -r requirements.txt || true
python -m pip install mediapipe

# Step 4: Solve dependency conflict
utils_dir=~/miniconda3/envs/video/lib/python3.11/site-packages/skvideo/io/
mkdir -p $utils_dir
mv abstract.py ~/miniconda3/envs/video/lib/python3.11/site-packages/skvideo/io/ || true
mv ffmpeg.py ~/miniconda3/envs/video/lib/python3.11/site-packages/skvideo/io/ || true

echo "Setup completed successfully!"

/home/chaerim/miniconda3/envs/video/etc/conda