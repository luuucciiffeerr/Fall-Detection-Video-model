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

MODEL_PATH = "simple3dcnn_fall_v2.pth"
CLIP_LEN = 16
RESIZE = (112, 112)
DEFAULT_THRESHOLD = 0.5

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

def load_clip(video_path: str, clip_len: int = 16, resize=(112, 112), start_frame=0):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame >= total_frames:
        start_frame = max(0, total_frames - clip_len)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(clip_len):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    if len(frames) < clip_len:
        pad_frames = np.zeros((clip_len - len(frames), *resize, 3), dtype=np.float32)
        frames.extend(pad_frames)
    clip = torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2).float()
    return clip

def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }

def predict_single_video(model, video_path, device, num_clips=5, threshold=0.5, use_amp=True):
    try:
        info = get_video_info(video_path)
        total_frames = info['frame_count']
        if total_frames == 0:
            return None, None, "Video has 0 frames"

        positions = np.linspace(0, max(0, total_frames - CLIP_LEN), num_clips, dtype=int)
        probs = []
        with torch.no_grad():
            for pos in positions:
                clip = load_clip(video_path, CLIP_LEN, RESIZE, start_frame=pos)
                clip = clip.unsqueeze(0).to(device)
                if use_amp:
                    with autocast():
                        out = model(clip)
                else:
                    out = model(clip)
                p = torch.softmax(out, dim=1)[0, 1].cpu().item()
                probs.append(p)
        fall_prob = float(np.mean(probs))
        pred = "Fall" if fall_prob > threshold else "No Fall"
        return pred, fall_prob, None
    except Exception as e:
        return None, None, f"Error: {e}"

def main():
    epilog = """
Examples:
  python predict_fall.py video.mp4
  python predict_fall.py --folder ./test_videos/ --output results.csv
"""
    parser = argparse.ArgumentParser(description="Fall Detection Prediction v2.0",
                                     epilog=epilog)
    parser.add_argument("video", nargs="?", help="Single video path")
    parser.add_argument("--folder", help="Batch process folder of videos")
    parser.add_argument("--output", help="Save batch results to CSV")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Decision threshold (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--num-clips", type=int, default=5,
                        help="Number of clips per video (default 5)")
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        help="Force device (default auto)")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("FALL DETECTION PREDICTION v2.0")
    print("="*80)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found: {MODEL_PATH}")
        print("Run: python train_fall_pipeline.py")
        return

    print(f"\nLoading model: {MODEL_PATH}")
    model = Simple3DCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded!")

    results = []

    if args.video:
        if not os.path.exists(args.video):
            print(f"ERROR: video not found: {args.video}")
            return
        print(f"\n{'='*80}")
        print(f"Processing: {args.video}")
        print(f"{'='*80}")
        pred, prob, err = predict_single_video(
            model, args.video, device,
            num_clips=args.num_clips,
            threshold=args.threshold
        )
        if err:
            print("ERROR:", err)
        else:
            info = get_video_info(args.video)
            print("\nRESULTS:")
            print(f"  Prediction:  {pred}")
            print(f"  Fall prob:   {prob:.1%}")
            print(f"  Confidence:  {max(prob, 1-prob):.1%}")
            print(f"  Frames:      {info['frame_count']}, duration {info['duration']:.1f}s")
    elif args.folder:
        video_files = []
        for ext in ("*.mp4", "*.avi", "*.mov"):
            video_files += list(Path(args.folder).glob(f"**/{ext}"))
        if not video_files:
            print(f"ERROR: no videos in {args.folder}")
            return
        print(f"\nFound {len(video_files)} videos\n")
        for vp in tqdm(video_files, desc="Processing", unit="video"):
            pred, prob, err = predict_single_video(
                model, str(vp), device,
                num_clips=args.num_clips,
                threshold=args.threshold
            )
            if err:
                results.append({"filename": vp.name,
                                "prediction": "ERROR",
                                "fall_probability": "N/A"})
            else:
                info = get_video_info(str(vp))
                results.append({"filename": vp.name,
                                "prediction": pred,
                                "fall_probability": f"{prob:.4f}",
                                "frames": info["frame_count"],
                                "duration_sec": f"{info['duration']:.2f}"})
        print("\nBATCH SUMMARY")
        fall_n = sum(1 for r in results if r["prediction"] == "Fall")
        nofall_n = sum(1 for r in results if r["prediction"] == "No Fall")
        err_n = sum(1 for r in results if r["prediction"] == "ERROR")
        total = len(results)
        print(f"  Total:    {total}")
        print(f"  Falls:    {fall_n} ({fall_n/total:.1%})")
        print(f"  No Falls: {nofall_n} ({nofall_n/total:.1%})")
        print(f"  Errors:   {err_n}")
        if args.output:
            pd.DataFrame(results).to_csv(args.output, index=False)
            print(f"\nSaved CSV: {args.output}")
    else:
        parser.print_help()

    print("\n" + "="*80)
    print("Done!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
