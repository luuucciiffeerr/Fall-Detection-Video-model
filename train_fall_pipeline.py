#v1.2.01
import os
#from pyexpat import model
import zipfile
import cv2
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1000"  # FFmpeg first
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"   # GStreamer last/disabled
cv2.setLogLevel(3)  # quiet OpenCV
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
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import logging
# Suppress OpenCV GStreamer warnings
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['GSTREAMER_DEBUG'] = '0'

logging.getLogger("cv2").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# ============ CONFIG ============
ZIP_PATH = "archive.zip"
EXTRACT_ROOT = "fall_dataset"
VIDEOS_CSV = "videos_info.csv"
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
MODEL_PATH = "simple3dcnn_fall_v2.pth"
CHECKPOINT_PATH = "simple3dcnn_fall_checkpoint.pth"
METRICS_PLOT = "training_metrics_v2.png"
CONFUSION_PLOT = "confusion_matrix_v2.png"
MISCLASSIFIED_CSV = "misclassified_samples.csv"

CLIP_LEN = 16
RESIZE = (112, 112)
BATCH_SIZE = 8          # bigger batch for RTX 3070
NUM_EPOCHS = 50
LR = 1e-4
NUM_WORKERS = 5        # was originally 4
USE_AMP = True          # mixed precision

# ============ GPU INFO ============
def print_gpu_info():
    print("\n" + "="*80)
    print("🖥️  GPU OPTIMIZATION INFO")
    print("="*80)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        print(f"✓ CUDA available: {n_gpus} GPU(s) detected")
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} | {gpu_mem:.1f} GB VRAM")
        print("\n⚡ Optimizations enabled:")
        print(f"  ✓ Batch size: {BATCH_SIZE}")
        print(f"  ✓ Data workers: {NUM_WORKERS}")
        print(f"  ✓ Pin memory: True")
        print(f"  ✓ AMP (Mixed Precision): {USE_AMP}")
        print(f"  ✓ non_blocking transfer: True")
        print(f"\n📊 Training: {NUM_EPOCHS} epochs, {BATCH_SIZE} batch, {LR} LR")
        print("   Expected time: ~2-3 hours on RTX 3070")
    else:
        device = torch.device("cpu")
        print("⚠️  No CUDA available - using CPU (very slow)")
    print("="*80 + "\n")
    return device

# device = print_gpu_info()

# ============ DATA HELPERS ============
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
                elif "No_Fall" in parts:
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
    print("✅ Splits created:")
    print(f"Train: {len(train_df)} ({train_df['label'].mean():.1%} fall)")
    print(f"Test:  {len(test_df)} ({test_df['label'].mean():.1%} fall)")

# ============ AUGMENTED CLIP LOADING ============
def load_clip(video_path: str, clip_len: int = 16, resize=(112, 112), augment=False):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if augment and total_frames > clip_len:
        start_frame = np.random.randint(0, total_frames - clip_len)
    else:
        start_frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(clip_len):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if augment:
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            frame = frame * contrast * brightness
            frame = np.clip(frame, 0, 255)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) < clip_len:
        pad_frames = np.zeros((clip_len - len(frames), *resize, 3), dtype=np.float32)
        frames.extend(pad_frames)

    clip = np.array(frames)
    if augment and np.random.rand() > 0.5:
        clip = np.ascontiguousarray(np.fliplr(clip))
    return clip

class FallVideoDataset(Dataset):
    def __init__(self, csv_file: str, clip_len: int = 16, augment=False):
        self.df = pd.read_csv(csv_file)
        self.clip_len = clip_len
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # try a few nearby indices if loading fails
        orig_idx = idx
        for _ in range(3):
            row = self.df.iloc[idx]
            path = row["path"]
            label_val = int(row["label"])

            try:
                clip_np = load_clip(path, self.clip_len, augment=self.augment)
                if clip_np is None:
                    raise ValueError("Empty clip returned")

                clip = torch.from_numpy(clip_np).permute(3, 0, 1, 2).float()
                label = torch.tensor(label_val, dtype=torch.long)
                return clip, label

            except Exception as e:
                print(f"[WARN] failed to load {path}: {e}")
                # move to next sample
                idx = (idx + 1) % len(self.df)

        # if all retries failed, either:
        # 1) raise cleanly (recommended for debugging)
        raise RuntimeError(f"Could not load any clip for original idx {orig_idx}")
        # or 2) return a dummy tensor with random noise / zeros and the label

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# ============ MODEL (BN + GLOBAL POOL) ============
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, (3,3,3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32, 64, (3,3,3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2))
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============ TRAIN / EVAL ============
def train_epoch(model, loader, optimizer, criterion, device, scaler, train=True, use_amp=True, epoch=None, num_epochs=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    phase = "Train" if train else "Test"

    if epoch is not None and num_epochs is not None:
        desc = f"{phase} E{epoch+1}/{num_epochs}"
    else:
        desc = phase

    pbar = tqdm(loader, desc=desc, leave=True)
    for batch in pbar:
        
        if batch is None:
            continue
        clips, labels = [t.to(device, non_blocking=True) for t in batch]
        if train:
            optimizer.zero_grad()
        if train and use_amp:
            with autocast():
                outputs = model(clips)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/max(total,1):.3f}"})
    return total_loss/max(total,1), correct/max(total,1)

def evaluate_model(model, test_loader, device, return_preds=False, epoch=None, num_epochs=None):
    model.eval()
    all_preds, all_labels = [], []

    if epoch is not None and num_epochs is not None:
        desc = f"Eval  E{epoch+1}/{num_epochs}"
    else:
        desc = "Evaluating"

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=desc, leave=False):
            if batch is None:
                continue
            clips, labels = [t.to(device, non_blocking=True) for t in batch]
            outputs = model(clips)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print("\n📊 Test Results:")
    print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
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
    if return_preds:
        return acc, f1, all_preds, all_labels
    return acc, f1

# ============ MAIN ============
def main():
    print("🚀 Fall Detection Training Pipeline v2.0 (RTX 3070 Optimized)")
    print("="*80)
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Put {ZIP_PATH} in this folder")
    data_root = ensure_unzipped(ZIP_PATH, EXTRACT_ROOT)
    build_videos_csv(data_root, VIDEOS_CSV)
    split_datasets(VIDEOS_CSV)

    train_ds = FallVideoDataset(TRAIN_CSV, augment=True)
    test_ds  = FallVideoDataset(TEST_CSV, augment=False)

    train_loader = DataLoader(train_ds, BATCH_SIZE, True,
                              collate_fn=collate_fn,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, False,
                              collate_fn=collate_fn,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

    print(f"\nTrain batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    device = print_gpu_info()


    model = Simple3DCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler() if USE_AMP else None

    start_epoch = 0

    # Prefer full checkpoint for resume
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if USE_AMP and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"➡️  Continuing from epoch {start_epoch+1}")
    elif os.path.exists(MODEL_PATH):
        # Fallback: only weights, fresh optimizer
        print(f"🔄 Loading existing model weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        start_epoch = 0  # or set manually if you know last epoch
    else:
        print("🆕 Starting from scratch")

    print(f"\n🎯 Training from epoch {start_epoch+1} to {NUM_EPOCHS}...")

    
    
    
    
    
    train_losses, train_accs, test_accs, test_f1s = [], [], [], []
    best_acc = 0
    best_epoch = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*80}")

        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device, scaler, train=True, use_amp=USE_AMP, epoch=epoch, num_epochs=NUM_EPOCHS)

        train_losses.append(tl)
        train_accs.append(ta)

        test_acc, test_f1 = evaluate_model(model, test_loader, device, return_preds=False, epoch=epoch, num_epochs=NUM_EPOCHS)
        print(f"\n📣 Epoch {epoch+1}/{NUM_EPOCHS} summary:")
        print(f"   Train Loss: {tl:.4f}")
        print(f"   Train Acc : {ta:.4f}")
        print(f"   Test  Acc : {test_acc:.4f}")
        print(f"   Test  F1  : {test_f1:.4f}")
        
        test_accs.append(test_acc)
        test_f1s.append(test_f1)

    

        # Save best model weights
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✨ New best model! Accuracy: {test_acc:.4f}")
        print(f"   Best Acc so far: {best_acc:.4f} (epoch {best_epoch})")

        # Save full checkpoint for resume
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if USE_AMP:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        torch.save(ckpt, CHECKPOINT_PATH)

        scheduler.step()

            
            
        

    print(f"\n🏆 Best model from epoch {best_epoch} with accuracy {best_acc:.4f}")
    print(f"💾 Model saved: {MODEL_PATH}")

    print("\n" + "="*80)
    print("FINAL EVALUATION WITH MISCLASSIFIED ANALYSIS")
    print("="*80)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    final_acc, final_f1, all_preds, all_labels = evaluate_model(model, test_loader, device, return_preds=True)

    print("\n📋 Training curves...")
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train Loss'); plt.grid(True, alpha=0.3); plt.legend()
    plt.subplot(1,3,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy'); plt.grid(True, alpha=0.3); plt.legend()
    plt.subplot(1,3,3)
    plt.plot(test_f1s, label='Test F1')
    plt.xlabel('Epoch'); plt.ylabel('F1'); plt.title('F1 Score'); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(METRICS_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Training charts saved: {METRICS_PLOT}")

    print("\n" + "="*80)
    print("✅ Pipeline complete!")
    print("="*80)
    print(f"\nFinal Metrics:\n  Accuracy: {final_acc:.4f}\n  F1 Score: {final_f1:.4f}\n  Best Epoch: {best_epoch}")

if __name__ == "__main__":
    main()
