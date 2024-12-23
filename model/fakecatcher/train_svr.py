import yaml
from tqdm import tqdm
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.feature.feature_extractor import FeatureExtractor
from svr.svr_model import SVRModel
from utils.logging import setup_logging

def main():
    data = joblib.load('features.pkl')

    # Set up logging
    global logger
    logger = setup_logging()

    video_features = data['features']  # Assumed to be a list of numpy arrays
    video_labels = data['labels']     # Assumed to be a list of labels

    valid_features = []
    valid_labels = []

    # nan 제거
    for features, label in zip(video_features, video_labels):
        if not np.any(np.isnan(features)):  # Check if there are no NaN values in features
            valid_features.append(features)
            valid_labels.append(label)


    # Combine all features and labels
    all_features = np.vstack(valid_features)
    all_labels = np.hstack([[label] * len(features) for label, features in zip(valid_labels, valid_features)])
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)

    # Train SVR model
    model = SVRModel()
    save_interval = 100
    for i in range(0, len(X_train), save_interval):
        batch_X = X_train[i:i+save_interval]
        batch_y = y_train[i:i+save_interval]
        model.train(batch_X, batch_y)
        model.save_model(f'svr_model_{i}.pkl')
        logger.info(f"Model saved at iteration {i}.")

if __name__=='__main__':
    main()