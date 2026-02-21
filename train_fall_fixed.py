"""
Fall Detection Training Pipeline v3.0
R(2+1)D-18 + Class Weighting + Smart Sampling
FIXED: CSV column names match your actual data (num_frames not numframes)
"""

import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# Suppress warnings
logging.getLogger("cv2").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

ZIPPATH = "archive.zip"
EXTRACTROOT = "falldataset"
VIDEOSCSV = "videos_info.csv"
TRAINCSV = "train.csv"
TESTCSV = "test.csv"

MODELPATH = "r2plus1d_fall_v3.pth"
CHECKPOINTPATH = "r2plus1d_fall_checkpoint.pth"
METRICSPLOT = "training_metrics_v3.png"
CONFUSIONPLOT = "confusion_matrix_v3.png"

# Hyperparameters optimized for 12-hour window
CLIPLEN = 16  # frames per clip
RESIZE = (112, 112)  # input resolution
BATCHSIZE = 16  # increased from 8 to utilize RTX 3070 fully
NUMEPOCHS = 12  # reduced from 50, with early stopping
LR = 3e-4  # increased from 1e-4 for faster convergence
NUMWORKERS = 8  # increased from 5 for data loading speed
USEAMP = True  # mixed precision for speed
PATIENCE = 4  # early stopping patience (epochs)

# ============================================================================
# GPU UTILITIES
# ============================================================================

def print_gpu_info():
    """Print GPU configuration and optimization status."""
    print("=" * 80)
    print("GPU OPTIMIZATION INFO")
    print("=" * 80)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA available: {num_gpus} GPU(s) detected")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} | {gpu_mem:.1f} GB VRAM")
        
        print("\nOptimizations enabled:")
        print(f"  • Batch size: {BATCHSIZE}")
        print(f"  • Data workers: {NUMWORKERS}")
        print(f"  • Pin memory: True")
        print(f"  • AMP (Mixed Precision): {USEAMP}")
        print(f"  • Non-blocking transfers: True")
        print(f"  • Training: {NUMEPOCHS} epochs (early stop at {PATIENCE})")
        print(f"  • Expected time: 4-8 hours on RTX 3070")
        
    else:
        device = torch.device("cpu")
        print("⚠ No CUDA available - using CPU (very slow)")
    
    print("=" * 80)
    return device


# ============================================================================
# DATA PREPARATION
# ============================================================================

def ensure_unzipped(zippath: str, extractroot: str) -> str:
    """Extract zip if not already extracted."""
    if os.path.exists(extractroot):
        print(f"✓ Using existing extracted folder: {extractroot}")
        return extractroot
    
    os.makedirs(extractroot, exist_ok=True)
    print(f"📦 Extracting {zippath}...")
    with zipfile.ZipFile(zippath, "r") as zf:
        zf.extractall(extractroot)
    print("✓ Extraction complete")
    return extractroot


def build_videos_csv(dataroot: str, csvpath: str) -> str:
    """Scan directory and build CSV with video metadata."""
    if os.path.exists(csvpath):
        print(f"✓ Using existing video CSV: {csvpath}")
        return csvpath
    
    videopaths = []
    fallcount = nofallcount = 0
    
    print("📹 Scanning videos...")
    for root, _, files in tqdm(os.walk(dataroot), desc="Folders"):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                fullpath = os.path.join(root, file)
                parts = fullpath.split(os.sep)
                
                label = None
                if "Fall" in parts:
                    label = 1
                    fallcount += 1
                elif "NoFall" in parts or "No_Fall" in parts:
                    label = 0
                    nofallcount += 1
                
                if label is not None:
                    videopaths.append((fullpath, label))
    
    print(f"✓ Found {len(videopaths)} videos ({fallcount} Fall, {nofallcount} NoFall)")
    
    # Extract metadata
    data = []
    print("📊 Reading metadata...")
    for videopath, label in tqdm(videopaths, desc="Videos"):
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = framecount / fps if fps > 0 else 0
        cap.release()
        
        filename = os.path.basename(videopath)
        data.append([
            filename, videopath, framecount, fps, width, height,
            round(duration, 2), label
        ])
    
    # CRITICAL: Use column names matching your existing CSVs
    df = pd.DataFrame(data, columns=[
        "filename", "path", "num_frames", "fps", "width", "height", 
        "duration_sec", "label"
    ])
    df.to_csv(csvpath, index=False)
    
    print(f"✓ Saved {csvpath} | Class balance: {df['label'].value_counts().to_dict()}")
    return csvpath


def split_datasets(videoscsv: str) -> None:
    """Split into train/test with stratification."""
    if all(os.path.exists(p) for p in [TRAINCSV, TESTCSV]):
        print(f"✓ Using existing splits: {TRAINCSV}, {TESTCSV}")
        return
    
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(videoscsv)
    traindf, testdf = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    
    traindf.to_csv(TRAINCSV, index=False)
    testdf.to_csv(TESTCSV, index=False)
    
    print(f"✓ Splits created")
    print(f"  • Train: {len(traindf)} ({traindf['label'].mean():.1%} Fall)")
    print(f"  • Test:  {len(testdf)} ({testdf['label'].mean():.1%} Fall)")


# ============================================================================
# VIDEO CLIP LOADING (Smart Sampling for Fall Videos)
# ============================================================================

def load_clip(
    videopath: str,
    cliplen: int = 16,
    resize: tuple = (112, 112),
    startframe: int = 0,
    augment: bool = False
) -> np.ndarray:
    """
    Load a single clip from video.
    
    Args:
        videopath: path to video file
        cliplen: number of frames to extract
        resize: (height, width) for resizing
        startframe: starting frame index
        augment: apply random brightness/contrast if True
    
    Returns:
        clip: (T, H, W, C) as float32 in [0, 1]
    """
    cap = cv2.VideoCapture(videopath)
    totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate startframe
    if startframe > totalframes - cliplen:
        startframe = max(0, totalframes - cliplen)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
    
    frames = []
    for _ in range(cliplen):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Random brightness/contrast augmentation
        if augment:
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            frame = frame * contrast * brightness
            frame = np.clip(frame, 0, 255)
        
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    
    cap.release()
    
    # Pad if needed
    if len(frames) < cliplen:
        padframes = np.zeros((cliplen - len(frames), *resize, 3), dtype=np.float32)
        frames.extend(padframes)
    
    clip = np.array(frames)  # (T, H, W, 3)
    
    # Random horizontal flip
    if augment and np.random.rand() > 0.5:
        clip = np.ascontiguousarray(np.fliplr(clip))
    
    return clip


# ============================================================================
# DATASET CLASS (Smart Sampling)
# ============================================================================

class FallVideoDataset(Dataset):
    """
    Dataset for fall detection videos with smart sampling.
    For Fall videos, sample from later half to ensure fall is in clip.
    """
    
    def __init__(self, csvfile: str, cliplen: int = 16, augment: bool = False):
        self.df = pd.read_csv(csvfile)
        self.cliplen = cliplen
        self.augment = augment
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        orig_idx = idx
        
        # Try up to 3 times (robust to corrupted frames)
        for attempt in range(3):
            try:
                row = self.df.iloc[idx]
                path = row["path"]
                labelval = int(row["label"])
                
                # CRITICAL FIX: Use 'num_frames' not 'numframes'
                totalframes = int(row["num_frames"])
                
                # Smart sampling for Fall videos
                if labelval == 1 and totalframes > self.cliplen:
                    # Fall: sample from latter half of video
                    start_min = totalframes // 2
                    start_max = max(start_min, totalframes - self.cliplen)
                    startframe = np.random.randint(start_min, start_max + 1)
                else:
                    # No_Fall: uniform random or 0 if short
                    if self.augment and totalframes > self.cliplen:
                        startframe = np.random.randint(0, totalframes - self.cliplen + 1)
                    else:
                        startframe = 0
                
                clipnp = load_clip(
                    path, self.cliplen, augment=self.augment, startframe=startframe
                )
                
                if clipnp is None:
                    raise ValueError("Empty clip returned")
                
                # Convert to (C, T, H, W) for 3D CNN
                clip = torch.from_numpy(clipnp.permute(3, 0, 1, 2)).float()
                label = torch.tensor(labelval, dtype=torch.long)
                
                return clip, label
            
            except Exception as e:
                idx = (idx + 1) % len(self.df)
                if attempt == 2:
                    raise RuntimeError(
                        f"Could not load any clip for original idx {orig_idx}: {e}"
                    )
        
        raise RuntimeError("Unexpected error in data loading")


def collate_fn(batch):
    """Custom collate to handle None batches."""
    batch = [b for b in batch if b is not None]
    if batch:
        return torch.utils.data.dataloader.default_collate(batch)
    return None


# ============================================================================
# MODEL: R(2+1)D-18 with Custom Head
# ============================================================================

class FallDetectionModel(nn.Module):
    """
    R(2+1)D-18 pretrained on Kinetics-400, fine-tuned for binary fall detection.
    Only final FC layer is trainable by default (transfer learning).
    """
    
    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        
        # Load pretrained R(2+1)D-18
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.backbone = r2plus1d_18(weights=weights)
        
        # Replace final FC layer for binary classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Optionally freeze backbone for faster training
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) - batch of video clips
        
        Returns:
            logits: (B, 2) - class logits
        """
        return self.backbone(x)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    epoch: int,
    num_epochs: int,
    use_amp: bool = True
):
    """
    Train for one epoch.
    
    Returns:
        avg_loss: average loss
        accuracy: batch-level accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Train E{epoch+1}/{num_epochs}", leave=True)
    
    for batch in pbar:
        if batch is None:
            continue
        
        clips, labels = batch
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward with mixed precision
        if use_amp:
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
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/max(total,1):.3f}"})
    
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_model(
    model,
    loader,
    device,
    return_preds: bool = False,
    epoch: int = None,
    num_epochs: int = None
):
    """
    Evaluate model on validation/test set.
    
    Returns:
        accuracy, f1_score, (optional: predictions, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    if epoch is not None and num_epochs is not None:
        desc = f"Eval E{epoch+1}/{num_epochs}"
    else:
        desc = "Evaluating"
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            if batch is None:
                continue
            
            clips, labels = batch
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(clips)
            preds = outputs.argmax(1).cpu()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    # Print detailed report
    print("\n📊 Test Results:")
    print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(all_labels, all_preds, target_names=["No_Fall", "Fall"]))
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No_Fall", "Fall"], yticklabels=["No_Fall", "Fall"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(CONFUSIONPLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix saved: {CONFUSIONPLOT}")
    
    if return_preds:
        return acc, f1, all_preds, all_labels
    return acc, f1


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("Fall Detection Training Pipeline v3.0")
    print("R(2+1)D-18 + Class Weighting + Smart Sampling")
    print("=" * 80 + "\n")
    
    # GPU setup
    device = print_gpu_info()
    
    # Data preparation
    if not os.path.exists(ZIPPATH):
        print(f"ℹ️  {ZIPPATH} not found - using existing extracted data")
    else:
        dataroot = ensure_unzipped(ZIPPATH, EXTRACTROOT)
        build_videos_csv(dataroot, VIDEOSCSV)
    
    split_datasets(VIDEOSCSV)
    
    # Load datasets
    print(f"\n📦 Loading datasets...")
    trainds = FallVideoDataset(TRAINCSV, augment=True)
    testds = FallVideoDataset(TESTCSV, augment=False)
    
    trainloader = DataLoader(
        trainds, BATCHSIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUMWORKERS,
        pin_memory=True
    )
    testloader = DataLoader(
        testds, BATCHSIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUMWORKERS,
        pin_memory=True
    )
    
    print(f"✓ Train batches: {len(trainloader)}")
    print(f"✓ Test batches: {len(testloader)}")
    
    # Compute class weights for imbalanced data
    print(f"\n⚖️  Computing class weights...")
    train_df = pd.read_csv(TRAINCSV)
    label_counts = train_df["label"].value_counts().sort_index()
    num_no_fall = label_counts[0]
    num_fall = label_counts[1]
    total = num_no_fall + num_fall
    
    class_weights = torch.tensor(
        [total / num_no_fall, total / num_fall],
        dtype=torch.float32,
        device=device,
    )
    print(f"✓ Class weights: No_Fall={class_weights[0]:.3f}, Fall={class_weights[1]:.3f}")
    
    # Model, optimizer, loss
    print(f"\n🧠 Initializing model...")
    model = FallDetectionModel(num_classes=2, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model loaded | Trainable params: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUMEPOCHS)
    scaler = GradScaler() if USEAMP else None
    
    # Resume from checkpoint if exists
    start_epoch = 0
    if os.path.exists(CHECKPOINTPATH):
        print(f"\n📂 Resuming from checkpoint...")
        ckpt = torch.load(CHECKPOINTPATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if USEAMP and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"✓ Continuing from epoch {start_epoch}")
    elif os.path.exists(MODELPATH):
        print(f"\n📂 Loading existing model weights...")
        model.load_state_dict(torch.load(MODELPATH, map_location=device))
        print(f"✓ Model weights loaded")
    else:
        print(f"\n🆕 Starting from scratch")
    
    # Training loop with early stopping
    print(f"\n🚀 Training from epoch {start_epoch} to {NUMEPOCHS}...")
    
    train_losses = []
    train_accs = []
    test_accs = []
    test_f1s = []
    
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(start_epoch, NUMEPOCHS):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch+1}/{NUMEPOCHS}")
        print("=" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, criterion, device, scaler,
            epoch, NUMEPOCHS, use_amp=USEAMP
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        test_acc, test_f1 = evaluate_model(
            model, testloader, device, epoch=epoch, num_epochs=NUMEPOCHS
        )
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        
        # Logging
        print(f"\n✓ Train Loss: {train_loss:.4f}")
        print(f"✓ Train Acc:  {train_acc:.4f}")
        print(f"✓ Test Acc:   {test_acc:.4f}")
        print(f"✓ Test F1:    {test_f1:.4f}")
        
        # Save best model
        if test_f1 > best_f1 + 1e-4:
            best_f1 = test_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save(model.state_dict(), MODELPATH)
            print(f"✅ New best model! F1={test_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if USEAMP:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        torch.save(ckpt, CHECKPOINTPATH)
        
        scheduler.step()
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⛔ Early stopping triggered (patience={PATIENCE})")
            break
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    model.load_state_dict(torch.load(MODELPATH, map_location=device))
    final_acc, final_f1, all_preds, all_labels = evaluate_model(
        model, testloader, device, return_preds=True
    )
    
    # Plot training curves
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(test_accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(test_f1s, label="Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(METRICSPLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Training curves saved: {METRICSPLOT}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n📈 Final Metrics:")
    print(f"  • Accuracy: {final_acc:.4f}")
    print(f"  • F1 Score: {final_f1:.4f}")
    print(f"  • Best Epoch: {best_epoch}")
    print(f"  • Model: {MODELPATH}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
