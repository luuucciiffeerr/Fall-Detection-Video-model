"""
Fall Detection Inference v3.0
Uses R(2+1)D-18 model for batch and single video prediction
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELPATH = "r2plus1d_fall_v3.pth"
CLIPLEN = 16
RESIZE = (112, 112)
DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_CLIPS = 5


# ============================================================================
# MODEL ARCHITECTURE (Must match training)
# ============================================================================

class FallDetectionModel(nn.Module):
    """R(2+1)D-18 for fall detection (inference version)."""
    
    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        
        # Import here to avoid circular dependency
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.backbone = r2plus1d_18(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# VIDEO CLIP LOADING (Inference)
# ============================================================================

def load_clip_from_video(
    videopath: str,
    cliplen: int = 16,
    resize: tuple = (112, 112),
    startframe: int = 0
) -> np.ndarray:
    """
    Load a single clip from video without augmentation.
    
    Args:
        videopath: path to video file
        cliplen: number of frames to extract
        resize: (height, width) for resizing
        startframe: starting frame index
    
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
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    
    cap.release()
    
    # Pad if needed
    if len(frames) < cliplen:
        padframes = np.zeros((cliplen - len(frames), *resize, 3), dtype=np.float32)
        frames.extend(padframes)
    
    return np.array(frames)  # (T, H, W, 3)


def get_video_info(videopath: str) -> dict:
    """Extract video metadata."""
    cap = cv2.VideoCapture(videopath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = framecount / fps if fps > 0 else 0
    cap.release()
    
    return {
        "fps": fps,
        "framecount": framecount,
        "width": width,
        "height": height,
        "duration": duration,
    }


# ============================================================================
# INFERENCE
# ============================================================================

def predict_single_video(
    model,
    videopath: str,
    device,
    num_clips: int = 5,
    threshold: float = 0.5,
    use_amp: bool = True
) -> tuple:
    """
    Predict fall for a single video by averaging predictions from multiple clips.
    
    Args:
        model: trained model
        videopath: path to video
        device: torch device
        num_clips: number of clips to sample from video
        threshold: decision threshold (prob > threshold = Fall)
        use_amp: use automatic mixed precision
    
    Returns:
        (prediction, fall_probability, error_message)
    """
    try:
        info = get_video_info(videopath)
        totalframes = info["framecount"]
        
        if totalframes == 0:
            return None, None, "Video has 0 frames"
        
        # Sample clip positions uniformly across video
        positions = np.linspace(
            0, max(0, totalframes - CLIPLEN),
            num_clips, dtype=int
        )
        
        probs = []
        model.eval()
        
        with torch.no_grad():
            for pos in positions:
                # Load and preprocess clip
                clip = load_clip_from_video(
                    videopath, CLIPLEN, RESIZE, startframe=pos
                )
                
                # Convert to (C, T, H, W) for model
                clip = torch.from_numpy(clip.permute(3, 0, 1, 2)).float()
                clip = clip.unsqueeze(0).to(device)  # Add batch dimension
                
                # Inference with mixed precision if available
                if use_amp:
                    with autocast():
                        output = model(clip)
                else:
                    output = model(clip)
                
                # Get fall probability
                probs_raw = torch.softmax(output, dim=1)
                fall_prob = probs_raw[0, 1].cpu().item()
                probs.append(fall_prob)
        
        # Average predictions from all clips
        avg_fall_prob = float(np.mean(probs))
        prediction = "Fall" if avg_fall_prob > threshold else "No_Fall"
        
        return prediction, avg_fall_prob, None
    
    except Exception as e:
        return None, None, f"Error: {e}"


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_batch_videos(
    model,
    video_folder: str,
    device,
    num_clips: int = 5,
    threshold: float = 0.5,
    use_amp: bool = True
) -> list:
    """Process all videos in a folder."""
    results = []
    
    # Find all video files
    videofiles = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        videofiles.extend(list(Path(video_folder).glob(ext)))
    
    if not videofiles:
        print(f"⚠️  No videos found in {video_folder}")
        return results
    
    print(f"\n📹 Found {len(videofiles)} videos. Processing...")
    
    for videopath in tqdm(videofiles, desc="Processing", unit="video"):
        prediction, fall_prob, error = predict_single_video(
            model, str(videopath), device,
            num_clips=num_clips, threshold=threshold, use_amp=use_amp
        )
        
        if error:
            results.append({
                "filename": videopath.name,
                "prediction": "ERROR",
                "fall_probability": "N/A",
            })
        else:
            info = get_video_info(str(videopath))
            results.append({
                "filename": videopath.name,
                "prediction": prediction,
                "fall_probability": f"{fall_prob:.4f}",
                "frames": info["framecount"],
                "duration_sec": f"{info['duration']:.2f}",
            })
    
    return results


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection Prediction v3.0",
        epilog="Examples:\n"
               "  python predict_fall.py video.mp4\n"
               "  python predict_fall.py --folder ./videos --output results.csv\n"
    )
    
    parser.add_argument(
        "video", nargs="?",
        help="Single video path (optional if using --folder)"
    )
    parser.add_argument(
        "--folder",
        help="Batch process folder of videos"
    )
    parser.add_argument(
        "--output",
        help="Save batch results to CSV"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Decision threshold (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--num-clips", type=int, default=DEFAULT_NUM_CLIPS,
        help=f"Number of clips per video (default: {DEFAULT_NUM_CLIPS})"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"],
        help="Force device (default: auto)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FALL DETECTION PREDICTION v3.0")
    print("=" * 80)
    
    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n💻 Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    if not os.path.exists(MODELPATH):
        print(f"\n❌ Model not found: {MODELPATH}")
        print("Run python train_fall_v3.py first")
        return
    
    print(f"\n📦 Loading model: {MODELPATH}")
    model = FallDetectionModel(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODELPATH, map_location=device))
    model.eval()
    print("✓ Model loaded!")
    
    # Process
    if args.video:
        # Single video
        if not os.path.exists(args.video):
            print(f"\n❌ Video not found: {args.video}")
            return
        
        print(f"\n{'='*80}")
        print(f"Processing: {args.video}")
        print(f"{'='*80}")
        
        prediction, fall_prob, error = predict_single_video(
            model, args.video, device,
            num_clips=args.num_clips, threshold=args.threshold
        )
        
        if error:
            print(f"❌ ERROR: {error}")
        else:
            info = get_video_info(args.video)
            print(f"\n📊 Result:")
            print(f"  Prediction:      {prediction}")
            print(f"  Fall Probability: {fall_prob:.1%}")
            print(f"  Confidence:      {max(fall_prob, 1-fall_prob):.1%}")
            print(f"  Frames:          {info['framecount']}")
            print(f"  Duration:        {info['duration']:.1f}s")
    
    elif args.folder:
        # Batch processing
        if not os.path.exists(args.folder):
            print(f"\n❌ Folder not found: {args.folder}")
            return
        
        results = process_batch_videos(
            model, args.folder, device,
            num_clips=args.num_clips, threshold=args.threshold
        )
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        fall_count = sum(1 for r in results if r["prediction"] == "Fall")
        nofall_count = sum(1 for r in results if r["prediction"] == "No_Fall")
        error_count = sum(1 for r in results if r["prediction"] == "ERROR")
        total = len(results)
        
        print(f"  Total:     {total}")
        print(f"  Falls:     {fall_count} ({fall_count/max(total,1):.1%})")
        print(f"  No Falls:  {nofall_count} ({nofall_count/max(total,1):.1%})")
        print(f"  Errors:    {error_count}")
        
        # Save CSV if requested
        if args.output:
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")
    
    else:
        parser.print_help()
    
    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
