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
apt install cmake -y
pip install -r requirements.txt || true
python -m pip install mediapipe

# Step 4: Download additional resources
echo "Downloading additional resources..."
mkdir -p misc/
wget -nc http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O misc/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d misc/shape_predictor_68_face_landmarks.dat.bz2 || true
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O misc/20words_mean_face.npy || true

# Step 5: Create dataset directories
echo "Creating directories..."
mkdir -p data/video/ data/roi/ data/audio/

# Step 6: Fix dependency issue
echo "Fixing dependency issue..."
utils_dir=~/miniconda3/envs/video/lib/python3.11/site-packages/skvideo/io/
mkdir -p $utils_dir
mv utils/abstract.py $utils_dir || true
mv utils/ffmpeg.py $utils_dir || true

echo "Setup completed successfully!"