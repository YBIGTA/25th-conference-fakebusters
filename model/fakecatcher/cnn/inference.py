import yaml
import logging
import argparse
import sys
import os
import torch
import numpy as np

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn import CNN
from utils.roi import ROIProcessor
from ppg.ppg_map import PPG_MAP

def predict(video_path: str, config_path: str) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Logger setup (add terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Load video
    logger.info("Processing the video...")
    landmarker = ROIProcessor(video_path, config)
    transformed_frames, fps = landmarker.detect_with_map()

    # Load CNN model
    logger.info("Loading the model...")
    model_path = config["model_path"]
    w = int(config["fps_standard"] * config["seg_time_interval"])  # Set input data height

    model = CNN(w=w, input_channels=1).to(device)  # Initialize model
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # Load saved weights
    model.eval()  # Set to evaluation mode

    predictions = []
    for segment in transformed_frames:
        logger.info("Generating PPG map...")
        ppg_map = PPG_MAP(segment, fps, config).compute_map()
        
        logger.info("Predicting...")
        ppg_map = torch.tensor(ppg_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 64, w)
        
        with torch.no_grad():
            output = model(ppg_map)  # Model prediction
            score = output.item()  # Extract probability value

        logger.info(f"Prediction Score: {score:.4f}")
        predictions.append(score)

    # Final prediction probability (returns a value between 0 and 1)
    final_score = np.mean(predictions)
    logger.info(f"Final Predicted Score: {final_score:.4f}")

    return final_score
