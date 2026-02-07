"""
RAVDESS Emotion Recognition - PyTorch CNN Implementation
=========================================================
Complete training pipeline with CNN model for emotion recognition
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. CUSTOM DATASET CLASS
# ============================================================================

class RAVDESSDataset(Dataset):
    """Custom PyTorch Dataset for RAVDESS spectrograms"""
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array of spectrograms (N, n_mels, time_steps)
            y: numpy array of labels (N,)
            transform: optional transform to apply
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        spectrogram = self.X[idx]
        label = self.y[idx]
        
        # Add channel dimension (N, 1, n_mels, time_steps)
        spectrogram = spectrogram.unsqueeze(0)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label


# ============================================================================
# 2. CNN MODEL ARCHITECTURE
# ============================================================================

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.adaptive_pool(x)
        x = self.dropout4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        return x


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
    
    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement for {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, model_name='emotion_cnn'):
    """Complete training loop"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}\n")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
    
    # History
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            print(f"✓ Best model saved (Val Acc: {val_acc*100:.2f}%)")
        
        # Early stopping
        if early_stopping(model, val_loss):
            print(f"\n{early_stopping.status}")
            break
        
        print()
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"{'='*80}\n")
    
    return history


# ============================================================================
# 4. EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, test_loader, device, idx_to_emotion):
    """Evaluate model on test set"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*80}")
    print(f"\nTest Accuracy:      {accuracy*100:.2f}%")
    print(f"Macro F1 Score:     {macro_f1:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}\n")
    
    # Classification report
    print("Classification Report:")
    print("-" * 80)
    emotion_names = [idx_to_emotion[i] for i in range(len(idx_to_emotion))]
    print(classification_report(all_labels, all_preds, target_names=emotion_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, macro_f1, all_preds, all_labels, cm


def plot_confusion_matrix(cm, idx_to_emotion, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    
    plt.figure(figsize=(12, 10))
    emotion_names = [idx_to_emotion[i] for i in range(len(idx_to_emotion))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Emotion Recognition', fontsize=16, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.show()


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, [acc*100 for acc in history['train_acc']], 'b-', 
                label='Train Acc', linewidth=2)
    axes[1].plot(epochs, [acc*100 for acc in history['val_acc']], 'r-', 
                label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history saved to: {save_path}")
    plt.show()


# ============================================================================
# 5. PREDICTION FUNCTION
# ============================================================================

def predict_emotion(model, spectrogram, device, idx_to_emotion):
    """
    Predict emotion from a single spectrogram
    
    Args:
        model: trained PyTorch model
        spectrogram: numpy array (n_mels, time_steps)
        device: torch device
        idx_to_emotion: mapping from index to emotion name
    
    Returns:
        predicted emotion, confidence, all probabilities
    """
    model.eval()
    
    # Prepare input
    input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_emotion = idx_to_emotion[predicted.item()]
    confidence_score = confidence.item()
    all_probs = probabilities.cpu().numpy()[0]
    
    return predicted_emotion, confidence_score, all_probs


# ============================================================================
# 6. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print(" " * 20 + "RAVDESS EMOTION RECOGNITION")
    print(" " * 22 + "PyTorch CNN Training Pipeline")
    print("="*80 + "\n")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    DATA_PATH = './processed_ravdess'
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("GPU not available, using CPU\n")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("Loading preprocessed data...")
    print("-" * 80)
    
    X_train = np.load(f'{DATA_PATH}/X_train.npy')
    X_val = np.load(f'{DATA_PATH}/X_val.npy')
    X_test = np.load(f'{DATA_PATH}/X_test.npy')
    
    y_train = np.load(f'{DATA_PATH}/y_train.npy')
    y_val = np.load(f'{DATA_PATH}/y_val.npy')
    y_test = np.load(f'{DATA_PATH}/y_test.npy')
    
    with open(f'{DATA_PATH}/label_mapping.json', 'r') as f:
        emotion_to_idx = json.load(f)
    
    idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
    num_classes = len(emotion_to_idx)
    
    print(f"✓ Training set:   {X_train.shape}")
    print(f"✓ Validation set: {X_val.shape}")
    print(f"✓ Test set:       {X_test.shape}")
    print(f"✓ Number of emotions: {num_classes}")
    print(f"✓ Emotions: {list(emotion_to_idx.keys())}")
    print("-" * 80 + "\n")
    
    # ========================================================================
    # CREATE DATASETS AND DATALOADERS
    # ========================================================================
    
    print("Creating DataLoaders...")
    
    train_dataset = RAVDESSDataset(X_train, y_train)
    val_dataset = RAVDESSDataset(X_val, y_val)
    test_dataset = RAVDESSDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    print(f"✓ Test batches:  {len(test_loader)}\n")
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    print("Building CNN model...")
    
    model = EmotionCNN(num_classes=num_classes, input_channels=1)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model: {model.__class__.__name__}")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}\n")
    
    # ========================================================================
    # SETUP TRAINING
    # ========================================================================
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        model_name='emotion_cnn'
    )
    
    # ========================================================================
    # PLOT TRAINING HISTORY
    # ========================================================================
    
    plot_training_history(history, save_path='emotion_cnn_training_history.png')
    
    # ========================================================================
    # EVALUATE ON TEST SET
    # ========================================================================
    
    # Load best model
    model.load_state_dict(torch.load('best_emotion_cnn.pth'))
    
    test_acc, test_f1, test_preds, test_labels, cm = evaluate_model(
        model, test_loader, device, idx_to_emotion
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, idx_to_emotion, save_path='emotion_cnn_confusion_matrix.png')
    
    # ========================================================================
    # EXAMPLE PREDICTIONS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("EXAMPLE PREDICTIONS")
    print("="*80 + "\n")
    
    # Get 5 random test samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        spectrogram = X_test[idx]
        true_label = y_test[idx]
        
        predicted_emotion, confidence, all_probs = predict_emotion(
            model, spectrogram, device, idx_to_emotion
        )
        
        print(f"Sample {i+1}:")
        print(f"  True Emotion:      {idx_to_emotion[true_label]}")
        print(f"  Predicted Emotion: {predicted_emotion}")
        print(f"  Confidence:        {confidence*100:.2f}%")
        print(f"  Top 3 predictions:")
        top3_indices = np.argsort(all_probs)[::-1][:3]
        for rank, pred_idx in enumerate(top3_indices, 1):
            print(f"    {rank}. {idx_to_emotion[pred_idx]:12s}: {all_probs[pred_idx]*100:6.2f}%")
        print()
    
    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'emotion_to_idx': emotion_to_idx,
        'idx_to_emotion': idx_to_emotion,
        'test_accuracy': test_acc,
        'test_macro_f1': test_f1,
        'history': history
    }, 'emotion_cnn_final.pth')
    
    print(f"{'='*80}")
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nSaved files:")
    print(f"  ✓ best_emotion_cnn.pth (best model weights)")
    print(f"  ✓ emotion_cnn_final.pth (final checkpoint)")
    print(f"  ✓ emotion_cnn_training_history.png")
    print(f"  ✓ emotion_cnn_confusion_matrix.png")
    print(f"\nTest Results:")
    print(f"  Accuracy:    {test_acc*100:.2f}%")
    print(f"  Macro F1:    {test_f1:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()