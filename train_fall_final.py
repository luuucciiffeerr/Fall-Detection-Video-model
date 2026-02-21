import warnings
import os

# Suppress OpenCV GStreamer warnings
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['GSTREAMER_DEBUG'] = '0'

warnings.filterwarnings('ignore')

import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.models.video as video_models
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# ================================================================================
# CONFIG
# ================================================================================
BATCHSIZE = 16
NUMWORKERS = 8
EPOCHS = 12
LR = 0.001
CLIP_LEN = 16
IMG_SIZE = 112
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================================
# DATASET CLASS
# ================================================================================
class FallVideoDataset(Dataset):
    def __init__(self, csv_path, video_csv_path, dataset_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.video_df = pd.read_csv(video_csv_path).set_index('filename')
        self.dataset_folder = Path(dataset_folder)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label = row['label']

        # Get video info
        video_info = self.video_df.loc[filename]
        total_frames = int(video_info['num_frames'])  # FIXED: num_frames
        fps = int(video_info['fps'])

        # Build video path
        video_path = self.dataset_folder / filename / 'Raw_Video' / f'{filename}.mp4'

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load video with smart sampling
        clip = self._load_clip(str(video_path), total_frames, label)

        if clip is None:
            raise RuntimeError(f"Could not load any clip for original idx {idx}: Failed to read video")

        # Convert to tensor
        clip_transposed = np.transpose(clip, (3, 0, 1, 2))  # FIXED: transpose not permute
        clip_tensor = torch.from_numpy(clip_transposed).float()

        return clip_tensor, label

    def _load_clip(self, video_path, total_frames, label, max_attempts=3):
        """Load video clip with smart sampling"""
        for attempt in range(max_attempts):
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    cap.release()
                    continue

                # Smart sampling: Falls from latter half
                if label == 1:  # Fall
                    start_frame = max(0, int(total_frames * 0.5))
                    end_frame = total_frames
                else:  # No_Fall - uniform random
                    start_frame = 0
                    end_frame = total_frames

                # Sample frame indices
                available_range = max(1, end_frame - start_frame)
                if available_range >= CLIP_LEN:
                    sampled_indices = np.sort(np.random.choice(
                        range(start_frame, end_frame), 
                        size=CLIP_LEN, 
                        replace=False
                    ))
                else:
                    sampled_indices = np.sort(np.random.choice(
                        range(start_frame, end_frame),
                        size=CLIP_LEN,
                        replace=True
                    ))

                frames = []
                for frame_idx in sampled_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret:
                        cap.release()
                        return None

                    # Resize to IMG_SIZE x IMG_SIZE
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

                cap.release()

                if len(frames) == CLIP_LEN:
                    return np.array(frames, dtype=np.uint8)

            except Exception as e:
                if attempt < max_attempts - 1:
                    continue
                return None

        return None


# ================================================================================
# MODEL
# ================================================================================
def get_model(num_classes=2):
    model = video_models.r2plus1d_18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    return model


# ================================================================================
# TRAINING FUNCTIONS
# ================================================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Train", leave=False)
    for batch in pbar:
        videos, labels = batch
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(videos)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.update(1)

    return total_loss / len(train_loader), correct / total


def eval_epoch(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Eval", leave=False)
        for batch in pbar:
            videos, labels = batch
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.update(1)

    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return total_loss / len(test_loader), correct / total, f1, all_preds, all_labels


# ================================================================================
# MAIN
# ================================================================================
def main():
    print("=" * 80)
    print(" " * 20 + "Fall Detection Training Pipeline v3.0 - FINAL FIX")
    print(" " * 15 + "R(2+1)D-18 + Class Weighting + Smart Sampling")
    print("=" * 80)

    # GPU info
    print("")
    print("=" * 80)
    print("GPU OPTIMIZATION INFO")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s) detected")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB VRAM")
    else:
        print("WARNING: CUDA not available - using CPU (will be VERY slow)")

    print("")
    print("Optimizations enabled:")
    print(f"  - Batch size: {BATCHSIZE}")
    print(f"  - Data workers: {NUMWORKERS}")
    print(f"  - Pin memory: True")
    print(f"  - AMP (Mixed Precision): True")
    print(f"  - Training: {EPOCHS} epochs (early stop at 4)")
    print(f"  - Expected time: 4-8 hours on RTX 3070")
    print("=" * 80)

    # Check paths
    train_csv = Path('train.csv')
    test_csv = Path('test.csv')
    video_csv = Path('videos_info.csv')
    dataset_folder = Path('falldataset')

    if not all([train_csv.exists(), test_csv.exists(), video_csv.exists()]):
        print("ERROR: Missing CSV files!")
        return

    if not dataset_folder.exists():
        print("ERROR: Missing dataset folder!")
        return

    print(f"Using existing extracted folder: {dataset_folder}")
    print(f"Using existing video CSV: {video_csv}")
    print(f"Using existing splits: {train_csv.name}, {test_csv.name}")

    # Load datasets
    print("")
    print("Loading datasets...")
    train_dataset = FallVideoDataset(str(train_csv), str(video_csv), str(dataset_folder))
    test_dataset = FallVideoDataset(str(test_csv), str(video_csv), str(dataset_folder))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=NUMWORKERS,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=NUMWORKERS,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Compute class weights - AGGRESSIVE for Fall detection
    print("")
    print("Computing class weights...")
    train_labels = train_dataset.df['label'].values
    class_counts = np.bincount(train_labels)

    # FIXED: AGGRESSIVE weights to force Fall learning
    weights = 1.0 / (class_counts / class_counts.sum())
    weights = weights / weights.sum() * len(weights)

    # Extra boost for Fall class
    weights[1] *= 3.0  # 3x penalty for missing Falls

    print(f"Class weights: No_Fall={weights[0]:.3f}, Fall={weights[1]:.3f}")

    # Model setup
    print("")
    print("Initializing model...")
    model = get_model(num_classes=2).to(DEVICE)

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded | Trainable params: {trainable_params:,}")

    # Loss and optimizer
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    print("")
    print("Starting from scratch")

    # Training loop
    print("")
    print(f"Training from epoch 0 to {EPOCHS}...")
    print("")

    best_f1 = 0
    patience = 0
    patience_limit = 4

    for epoch in range(EPOCHS):
        print("=" * 80)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 80)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )

        test_loss, test_acc, test_f1, all_preds, all_labels = eval_epoch(
            model, test_loader, criterion, DEVICE
        )

        print("")
        print("Test Results:")
        print(f"Accuracy: {test_acc:.4f} | F1 Score: {test_f1:.4f}")

        # Detailed report
        print("")
        print("Detailed Report:")
        report = classification_report(
            all_labels, all_preds,
            target_names=['No_Fall', 'Fall'],
            zero_division=0
        )
        print(report)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No_Fall', 'Fall'],
                    yticklabels=['No_Fall', 'Fall'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_v3.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved: confusion_matrix_v3.png")

        # Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_f1': best_f1,
        }
        torch.save(checkpoint, 'r2plus1d_fall_checkpoint.pth')

        print("")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc:  {train_acc:.4f}")
        print(f"Test Acc:   {test_acc:.4f}")
        print(f"Test F1:    {test_f1:.4f}")

        # Early stopping
        if test_f1 > best_f1:
            best_f1 = test_f1
            patience = 0
            torch.save(model.state_dict(), 'r2plus1d_fall_v3.pth')
            print(f"NEW BEST MODEL! F1={best_f1:.4f}")
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{patience_limit}")

        if patience >= patience_limit:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        print("")

    print("")
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model saved: r2plus1d_fall_v3.pth")
    print(f"Best F1 Score: {best_f1:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
