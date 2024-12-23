"""Evaluate pre-trained LipForensics model on various face forgery datasets"""

import argparse
from collections import defaultdict

import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm

import sys
sys.path.insert(0, "/root/25th-conference-fakebusters/Custom_LipForensics/LipForensics/data")
from transforms import NormalizeVideo, ToTensorVideo
from dataset_clips import SingleVideoClips
from samplers import ConsecutiveClipSampler
sys.path.insert(0, "/root/25th-conference-fakebusters/Custom_LipForensics/LipForensics/models")
from spatiotemporal_net import get_model
sys.path.insert(0, '/root/25th-conference-fakebusters/Custom_LipForensics/LipForensics')
from lipforensics_utils import get_files_from_split


def evaluate_lipforensics(
    cropped_mouths_path,
    weights_forgery_path="/root/25th-conference-fakebusters/Custom_LipForensics/LipForensics/models/weights/lipforensics_ff.pth",
    frames_per_clip=25,
    batch_size=8,
    device="cuda:0",
    grayscale=True,
    num_workers=4,
):
    """
    Evaluate LipForensics model on a single video.

    :param cropped_mouths_path: Path to the directory containing cropped mouths for a single video
    :param weights_forgery_path: Path to pretrained weights for forgery detection
    :param frames_per_clip: Number of frames per clip
    :param batch_size: Batch size for inference
    :param device: Device to put tensors on
    :param grayscale: Use grayscale frames if True, otherwise RGB
    :param num_workers: Number of workers for data loading
    :return: Prediction score (0: Real, 1: Fake)
    """
    # Load the pre-trained model
    model = get_model(weights_forgery_path=weights_forgery_path)

    # Transformation pipeline
    transform = Compose(
        [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
    )

    # Single video dataset
    dataset = SingleVideoClips(
        video_path=cropped_mouths_path,
        frames_per_clip=frames_per_clip,
        grayscale=grayscale,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Inference
    model.eval()
    logits_list = []
    with torch.no_grad():
        for clips, _ in tqdm(loader, desc="Running LipForensics Inference"):
            clips = clips.to(device)
            logits = model(clips, lengths=[frames_per_clip] * clips.shape[0])
            logits_list.append(logits)
    
    print(f"Number of clips in dataset: {len(logits_list)}")


    # Aggregate logits and compute prediction
    avg_logits = torch.mean(torch.cat(logits_list), dim=0)
    prediction = torch.sigmoid(avg_logits).item()
    return prediction

if __name__ == "__main__":
    prediction = evaluate_lipforensics(cropped_mouths_path='/root/cropped_mouths/008_990')
    print(f"Final Prediction: {prediction:.4f} (0: Real, 1: Fake)")