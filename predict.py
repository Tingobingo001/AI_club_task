import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import argparse
import os
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
# These must match the settings used to generate your X_train.npy
SAMPLE_RATE = 22050
DURATION = 3.0    # Duration in seconds
N_MELS = 128      # Mel bands (Standard for this architecture)
HOP_LENGTH = 512

# Path to the file saved by your training script
MODEL_PATH = 'emotion_cnn_final.pth'

# ==========================================
# 2. MODEL ARCHITECTURE (From your code)
# ==========================================
class EmotionCNN(nn.Module):
    """
    Convolutional Neural Network for emotion recognition from spectrograms
    """
    def __init__(self, num_classes=8, input_channels=1):
        super(EmotionCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.3)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Convolutional Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout4 = nn.Dropout(0.4)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout6 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        # Block 2
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        # Block 3
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        # Block 4
        x = self.dropout4(self.adaptive_pool(F.relu(self.bn4(self.conv4(x)))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = self.dropout5(F.relu(self.bn5(self.fc1(x))))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

# ==========================================
# 3. PREPROCESSING
# ==========================================
def preprocess_audio(file_path):
    """
    Loads audio and converts it to a Log-Mel Spectrogram.
    """
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # 2. Pad or Trim to fixed length
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]

        # 3. Generate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
        )
        
        # 4. Convert to Log Scale (dB) - Crucial for CNNs
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 5. Prepare for Model
        # Original shape: (n_mels, time)
        # Target shape: (Batch=1, Channel=1, n_mels, time)
        tensor = torch.FloatTensor(mel_spec_db)
        tensor = tensor.unsqueeze(0).unsqueeze(0) 
        
        return tensor

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
def predict(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Check for model file
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model '{MODEL_PATH}' not found.")
        print("Please ensure you have run the training script and generated 'emotion_cnn_final.pth'.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 1. Load the checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # 2. Extract Labels and Config
        # Your training script saves 'idx_to_emotion' inside the checkpoint
        idx_to_emotion = checkpoint.get('idx_to_emotion')
        if idx_to_emotion is None:
             # Fallback if dictionary key is missing
             print("Warning: Label mapping not found in checkpoint. Using default.")
             idx_to_emotion = {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fearful', 6:'disgust', 7:'surprised'}
        
        num_classes = len(idx_to_emotion)

        # 3. Initialize Model
        model = EmotionCNN(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 4. Process Audio
        input_tensor = preprocess_audio(file_path)
        if input_tensor is None: return
        input_tensor = input_tensor.to(device)

        # 5. Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get Top Prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_label = idx_to_emotion[predicted_idx.item()]
            confidence_val = confidence.item() * 100

        # 6. Print Results
        print("\n" + "="*40)
        print(f"File:       {os.path.basename(file_path)}")
        print(f"Prediction: \033[1m{predicted_label.upper()}\033[0m")
        print(f"Confidence: {confidence_val:.2f}%")
        print("="*40 + "\n")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a .wav file")
    parser.add_argument("file", type=str, help="Path to the audio file")
    
    args = parser.parse_args()
    predict(args.file)