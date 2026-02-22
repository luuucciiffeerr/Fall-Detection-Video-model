import warnings
import os

os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['GSTREAMER_DEBUG'] = '0'

warnings.filterwarnings('ignore')

import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from pathlib import Path

CLIP_LEN = 16
IMG_SIZE = 112
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELPATH = 'r2plus1d_fall_v3.pth'

def get_model(num_classes=2):
    model = video_models.r2plus1d_18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    return model

def load_video_clip(video_path, clip_len=CLIP_LEN, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return None

    frames = []
    while len(frames) < clip_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) < clip_len:
        print(f"Video too short: got {len(frames)} / {clip_len} frames")
        return None

    clip = np.array(frames[:clip_len], dtype=np.uint8)
    clip_transposed = np.transpose(clip, (3, 0, 1, 2))
    clip_tensor = torch.from_numpy(clip_transposed).float()
    return clip_tensor

def predict(video_path, model, device):
    clip = load_video_clip(video_path)
    if clip is None:
        return None, None
    clip = clip.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(clip)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence

def main():
    print("=" * 80)
    print("FALL DETECTION PREDICTION v3.0")
    print("=" * 80)
    print("")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print("")
    print(f"Loading model: {MODELPATH}")
    model = get_model(num_classes=2)
    state_dict = torch.load(MODELPATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    if len(sys.argv) < 2:
        print("Usage: python predict_fall.py <video_path>")
        return
    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return
    print("")
    print(f"Predicting on: {video_path}")
    pred_class, confidence = predict(video_path, model, DEVICE)
    if pred_class is None:
        print("Could not process video")
        return
    print("=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)
    class_name = "FALL" if pred_class == 1 else "NO FALL"
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == '__main__':
    main()
