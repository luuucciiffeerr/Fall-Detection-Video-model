#v1.1.01

import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============ PATHS ============
ZIP_PATH = "archive.zip"
EXTRACT_ROOT = "fall_dataset"
VIDEOS_CSV = "videos_info.csv"
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"  # No VAL_CSV anymore
MODEL_PATH = "simple3dcnn_fall.pth"
METRICS_PLOT = "training_metrics.png"
CONFUSION_PLOT = "confusion_matrix.png"

# Training config
CLIP_LEN = 16
RESIZE = (112, 112)
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4

def ensure_unzipped(zip_path: str, extract_root: str) -> str:
    if os.path.exists(extract_root):
        print(f"✓ Using existing {extract_root}")
        return extract_root
    
    os.makedirs(extract_root, exist_ok=True)
    print(f"📦 Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    print("✓ Extraction complete")
    return extract_root

def build_videos_csv(data_root: str, csv_path: str):
    """Walk dataset, find videos, extract metadata, save to CSV."""
    if os.path.exists(csv_path):
        print(f"✓ Using existing {csv_path}")
        return
    
    video_paths = []
    fall_count = nofall_count = 0
    
    print("🔍 Scanning videos...")
    for root, _, files in tqdm(os.walk(data_root), desc="Folders"):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                full_path = os.path.join(root, file)
                parts = full_path.split(os.sep)
                label = None
                if "Fall" in parts:
                    label = 1
                    fall_count += 1
                elif "No_Fall" in parts:  # Fixed: was "NoFall" but files use "No_Fall"
                    label = 0
                    nofall_count += 1
                if label is not None:
                    video_paths.append((full_path, label))
    
    print(f"📊 Found {len(video_paths)} videos: {fall_count} Fall, {nofall_count} No_Fall")
    
    data = []
    print("📹 Reading metadata...")
    for video_path, label in tqdm(video_paths, desc="Videos"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        filename = os.path.basename(video_path)
        data.append([filename, video_path, frame_count, fps, width, height, round(duration, 2), label])
    
    df = pd.DataFrame(data, columns=[
        "filename", "path", "num_frames", "fps", "width", "height", "duration_sec", "label"
    ])
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved {csv_path} ({len(df)} videos)")
    print("Class balance:", df['label'].value_counts().to_dict())

def split_datasets(videos_csv):
    """Create train/test CSVs only: 80% train, 20% isolated test."""
    if all(os.path.exists(p) for p in [TRAIN_CSV, TEST_CSV]):
        print("✓ Using existing splits")
        return
    
    df = pd.read_csv(videos_csv)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2,
        stratify=df['label'], 
        random_state=42
    )
    
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    print(f"✅ Splits created:")
    print(f"Train: {len(train_df)} ({train_df['label'].mean():.1%} fall)")
    print(f"Test:  {len(test_df)} ({test_df['label'].mean():.1%} fall)")

def load_clip(video_path: str, clip_len: int = 16, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < clip_len:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    
    if len(frames) < clip_len:
        pad_frames = np.zeros((clip_len - len(frames), *resize, 3), dtype=np.float32)
        frames.extend(pad_frames)
    
    return np.array(frames)

class FallVideoDataset(Dataset):
    def __init__(self, csv_file: str, clip_len: int = 16):
        self.df = pd.read_csv(csv_file)
        self.clip_len = clip_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_np = load_clip(row["path"], self.clip_len)
        if clip_np is None:
            return None
        
        clip = torch.from_numpy(clip_np).permute(3, 0, 1, 2).float()
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return clip, label

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, (3,3,3), padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32, 64, (3,3,3), padding=1), nn.ReLU(), nn.MaxPool3d((2,2,2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x): 
        x = self.features(x)
        return self.classifier(x)

def train_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    phase = "Train" if train else "Test"  # Fixed: no val anymore
    pbar = tqdm(loader, desc=phase)
    
    for batch in pbar:
        if batch is None: continue
        clips, labels = [t.to(device) for t in batch]
        
        if train: optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'acc': f"{correct/max(total,1):.3f}"
        })
    
    return total_loss/max(total,1), correct/max(total,1)

def evaluate_model(model, test_loader, device):
    """Full evaluation with metrics + confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if batch is None: continue
            clips, labels = [t.to(device) for t in batch]
            outputs = model(clips)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n📊 Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(all_labels, all_preds, target_names=['No_Fall', 'Fall']))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No_Fall', 'Fall'], yticklabels=['No_Fall', 'Fall'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CONFUSION_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Confusion matrix saved: {CONFUSION_PLOT}")
    
    return acc, f1

def main():
    print("🚀 Fall Detection Training Pipeline")
    
    # 1. Setup data
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Put {ZIP_PATH} in this folder")
    data_root = ensure_unzipped(ZIP_PATH, EXTRACT_ROOT)
    
    build_videos_csv(data_root, VIDEOS_CSV)
    split_datasets(VIDEOS_CSV)
    
    # 2. Loaders (FIXED: no val_loader)
    train_ds = FallVideoDataset(TRAIN_CSV)
    test_ds  = FallVideoDataset(TEST_CSV)
    
    train_loader = DataLoader(train_ds, BATCH_SIZE, True,  collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # 3. Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    model = Simple3DCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # 4. Train (FIXED: no val losses/accs, proper appends)
    print("\n🎯 Starting training...")
    train_losses, train_accs = [], []
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device, train=True)
        train_losses.append(tl)
        train_accs.append(ta)
        
        if (epoch + 1) % 5 == 0:
            print("🔍 Quick test eval:")
            test_acc, test_f1 = evaluate_model(model, test_loader, device)
    
    # 5. Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved: {MODEL_PATH}")
    
    # 6. Training chart (FIXED: only train metrics)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(METRICS_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Training charts saved: {METRICS_PLOT}")
    
    # 7. Final test evaluation
    print("\n🏆 FINAL TEST RESULTS:")
    final_acc, final_f1 = evaluate_model(model, test_loader, device)
    print("✅ Pipeline complete!")

if __name__ == "__main__":
    main()
