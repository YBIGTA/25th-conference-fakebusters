import torch
from torch.utils.data import DataLoader, random_split
from cnn import CNN
import yaml
import json
import numpy as np
from torch.utils.data import Dataset
import argparse
import logging

# Argument parsing
parser = argparse.ArgumentParser(description='Train CNN model for fakecatcher')
parser.add_argument('-c', '--config_path', required=True, help='Path to the config.yaml file')
parser.add_argument('-i', '--input_path', required=True, help='Path to the ppg_map_results_updated.json file')
parser.add_argument('-l', '--log_path', required=False, help='Path to save logs')
parser.add_argument('-o', '--output_path', required=False, help='Path to save the trained model')
args = parser.parse_args()

# Logging setup
log_path = args.log_path if args.log_path else 'train_cnn.log'
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),  # Save log to file
        logging.StreamHandler()         # Output to terminal
    ]
)
logging.info('Started training process')

# Settings
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')

# Data preparation
logging.info(f'Loading configuration from {args.config_path}')
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)
    fps_standard = config.get('fps_standard', 30)  # Default 30
    time_interval = config.get('seg_time_interval', 3)  # Default 3
    w = fps_standard * time_interval

logging.info(f'Loading data from {args.input_path}')
with open(args.input_path, 'r') as file:
    data = json.load(file)

# Extract PPG Maps and Labels
ppg_maps = []
labels = []
for item in data:
    ppg_maps.append(np.array(item['ppg_map']))  # Array of shape (64, w)
    labels.append(item['label'])               # 0 or 1

# Convert to numpy arrays
ppg_maps = np.array(ppg_maps)
labels = np.array(labels)     

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (np.array): Shape (2831, 64, 90)
            labels (np.array): Shape (2831,)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (data_sample, label)
        """
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(ppg_maps, labels)
logging.info(f"Dataset size: {len(dataset)}")
logging.info(f"Sample PPG map shape: {dataset[0][0].shape}")
logging.info(f"Sample label: {dataset[0][1]}")
train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
logging.info(f"Training dataset size: {train_size}")
logging.info(f"Validation dataset size: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model initialization
model = CNN(w=w, input_channels=1).to(device)
criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define training function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, output_path):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1).to(device)  # Convert to (batch, 1, 64, w)
            labels = labels.float().unsqueeze(1).to(device)  # Convert to (batch, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Binary classification after sigmoid
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1).to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_path if output_path else "cnn_model_state.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved to {best_model_path}")

# Execute training
output_model_path = args.output_path if args.output_path else "cnn_model_state.pth"
train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, output_model_path)

logging.info("Training completed.")
