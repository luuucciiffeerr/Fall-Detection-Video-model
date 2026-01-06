import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os

# Same model/dataset code
CLIP_LEN = 16
RESIZE = (112, 112)
MODEL_PATH = "simple3dcnn_fall.pth"

def load_clip(video_path: str, clip_len: int = 16, resize=(112, 112)):
    """Load 16-frame clip from video for prediction."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_idx = 0
    while len(frames) < clip_len and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_idx += 1
    
    cap.release()
    
    # Pad with black frames if too short
    if len(frames) < clip_len:
        pad_frames = np.zeros((clip_len - len(frames), *resize, 3), dtype=np.uint8)
        frames.extend(pad_frames)
    
    return np.array(frames)


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


class FallPredictor:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = Simple3DCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()
        print(f"✅ Model loaded: {model_path}")
    
    def predict_single_video(self, video_path: str, threshold: float = 0.5):
        """Predict fall/no-fall on one video. Returns (prediction, confidence)."""
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            print("Usage: python predict_fall.py path/to/real/video.mp4")
            return None, 0.0

        
        
        print(f"🎥 Analyzing: {video_path}")
        clip_np = load_clip(video_path)
        clip = torch.tensor(clip_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        clip = clip.to(self.device)
        
        with torch.no_grad():
            output = self.model(clip)
            probs = torch.softmax(output, dim=1)[0]
            fall_prob = probs[1].item()
            pred = "FALL ⚠️" if fall_prob > threshold else "No Fall ✅"
        
        print(f"Prediction: {pred}")
        print(f"Fall probability: {fall_prob:.3f}")
        return pred, fall_prob
    
    def predict_folder(self, folder_path: str, threshold: float = 0.5):
        """Batch predict all .mp4 videos in a folder."""
        videos = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
        if not videos:
            print("No .mp4 files found")
            return
        
        results = []
        for video in tqdm(videos, desc="Predicting"):
            video_path = os.path.join(folder_path, video)
            pred, prob = self.predict_single_video(video_path, threshold)
            results.append({'filename': video, 'prediction': pred, 'fall_prob': prob})
        
        df = pd.DataFrame(results)
        df.to_csv('predictions.csv', index=False)
        print(f"\n📊 Results saved: predictions.csv")
        print(df)
        return df


def main():
    parser = argparse.ArgumentParser(description="Fall Detection Predictor")
    parser.add_argument("video", nargs='?', help="Single video path")
    parser.add_argument("--folder", help="Folder of videos")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fall threshold (0-1)")
    args = parser.parse_args()
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Train first! Need {MODEL_PATH}")
    
    predictor = FallPredictor(MODEL_PATH)
    
    if args.video:
        predictor.predict_single_video(args.video, args.threshold)
    elif args.folder:
        predictor.predict_folder(args.folder, args.threshold)
    else:
        print("Usage:")
        print("  python predict_fall.py path/to/video.mp4")
        print("  python predict_fall.py --folder path/to/videos/")
        print("  python predict_fall.py video.mp4 --threshold 0.7")


if __name__ == "__main__":
    main()
