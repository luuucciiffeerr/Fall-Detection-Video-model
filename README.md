# 🎯 Folio Finder AI — Video-Based Fall Detection

[
[
[
[

A deep learning system for real-time fall detection from video using the **R(2+1)D-18** spatiotemporal convolutional neural network. Achieves **98.71% F1 score** on a custom dataset of ~7,000 video clips.

***

## 📊 Results at a Glance

| Metric | Score |
|--------|-------|
| **F1 Score** | 98.71% |
| **Accuracy** | 98.71% |
| **Precision (Fall)** | 99% |
| **Recall (Fall)** | 98% |
| **Inference Time** | <1 sec/video |

***

## 🧠 How It Works

The system uses **R(2+1)D-18**, a factored 3D CNN that decomposes spatiotemporal convolutions into separate spatial (2D) and temporal (1D) components. This architecture:

- Captures **body posture** (spatial) and **motion dynamics** (temporal) simultaneously
- Uses **transfer learning** from Kinetics-400 (pretrained on 400 action classes)
- Processes **16 frames** per clip at **112×112** resolution
- Outputs a binary classification: **Fall** or **No Fall**

### Pipeline Overview

```
Video Input → Frame Extraction (16 frames) → Resize (112×112) → R(2+1)D-18 → Softmax → Fall / No Fall
```

### Smart Temporal Sampling

- **Fall videos**: Frames sampled from the **latter half** (falls typically occur at the end)
- **No-Fall videos**: Frames sampled **uniformly** across the full duration

***

## 📁 Project Structure

```
Folio_Finder_AI/
├── train_fall_final.py           # Training pipeline
├── predict_fall.py               # Inference / prediction script
├── r2plus1d_fall_v3.pth          # Best model weights
├── r2plus1d_fall_checkpoint.pth  # Training checkpoint
├── videos_info.csv               # Full dataset catalog
├── train.csv                     # Training split
├── test.csv                      # Test split
├── confusion_matrix_v3.png       # Confusion matrix visualization
├── training_metrics_v3.png       # Training curves
├── requirements.txt              # Python dependencies
└── falldataset/
    ├── Fall/
    │   └── Raw_Video/            # Fall event clips
    └── Video/
        └── Raw_Video/            # No-fall activity clips
```

***

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- ~10 GB disk space for dataset

### Setup

```bash
# Clone the repository
git clone https://github.com/[your-username]/Folio_Finder_AI.git
cd Folio_Finder_AI

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
numpy>=1.24.0
```

***

## 🏋️ Training

### Quick Start

```bash
python train_fall_final.py
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 16 |
| Clip Length | 16 frames |
| Input Resolution | 112 × 112 |
| Max Epochs | 12 |
| Early Stopping | Patience 4 (F1-based) |
| Mixed Precision | Enabled (AMP) |
| Class Weights | No_Fall: 0.899, Fall: 3.304 |

### What Happens During Training

1. Loads `train.csv` / `test.csv` splits (or generates them from `videos_info.csv`)
2. Computes inverse-frequency class weights to handle class imbalance
3. Initializes R(2+1)D-18 with Kinetics-400 pretrained weights
4. Trains with weighted cross-entropy loss + mixed precision
5. Evaluates on test set after each epoch
6. Saves best model (by F1) and latest checkpoint
7. Generates confusion matrix and training curves

### Output Files

| File | Description |
|------|-------------|
| `r2plus1d_fall_v3.pth` | Best model weights |
| `r2plus1d_fall_checkpoint.pth` | Latest checkpoint (resumable) |
| `confusion_matrix_v3.png` | Test set confusion matrix |
| `training_metrics_v3.png` | Loss / Accuracy / F1 curves |

***

## 🔮 Inference

### Predict on a Single Video

```bash
python predict_fall.py "path/to/video.mp4"
```

### Example Output

```
Loading model...
Processing video: test_fall.mp4
Reading frames: 16 frames extracted
Prediction: Fall (confidence: 98.72%)
```

### Using in Python

```python
import torch
from torchvision.models.video import r2plus1d_18
import cv2
import numpy as np

# Load model
model = r2plus1d_18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load("r2plus1d_fall_v3.pth"))
model.eval()

# Process video (16 frames, 112x112, RGB)
# ... frame extraction logic ...

with torch.no_grad():
    output = model(video_tensor)
    pred = torch.softmax(output, dim=1)
    label = "Fall" if pred > pred[^1] else "No Fall"
    confidence = pred.max().item() * 100
    print(f"{label} ({confidence:.2f}%)")
```

***

## 📈 Dataset

### Overview

| Property | Value |
|----------|-------|
| Total clips | ~6,982 |
| Train set | ~5,584 (80%) |
| Test set | 1,398 (20%) |
| Classes | 2 (Fall, No_Fall) |
| Avg duration | 1–8 seconds |
| Frame rates | 15–120 FPS |
| Resolutions | 480p to 4K (normalized) |

### Sources

- **Public Kaggle datasets** (Fall Detection Dataset, Fall Video Dataset)
- **Original recordings** (smartphone, 1080p, 30fps — Sept 2024)
- **Research benchmarks** (SisFall-derived, multi-camera setups)

### Data Format

Each video is cataloged in `videos_info.csv`:

```csv
filename,path,num_frames,fps,width,height,duration_sec,label
example_fall.mp4,falldataset/Fall/Raw_Video/example_fall.mp4,57,30.0,1920,1080,1.9,0
example_nofall.mp4,falldataset/Video/Raw_Video/example_nofall.mp4,91,30.0,1100,1080,3.0,1
```

> **Note**: Label `0` = Fall, Label `1` = No_Fall

***

## 🏆 Model Comparison

| Method | Type | F1 / Accuracy | Hardware |
|--------|------|--------------|----------|
| **R(2+1)D-18 (Ours)** | **Video** | **98.71%** | **RTX 3070** |
| YOLOv8 + Transformer | Video | mAP 99.55% | High-end GPU |
| 4S-3DCNN | Video | 99.03% | Multi-GPU |
| CNN-LSTM | Video + Sensor | 96.4% | GPU |
| DSCS | Sensor only | 99.32% | CPU |
| Random Forest | Sensor only | 97.47% | CPU |
| LSTM | Sensor only | 80.0% | CPU |

***

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, torchvision
- **Video Processing**: OpenCV
- **Data Management**: pandas, NumPy
- **Evaluation**: scikit-learn
- **Visualization**: matplotlib
- **Training Optimization**: CUDA AMP (mixed precision), DataLoader with pin_memory

***

## 📝 Training Logs

<details>
<summary>Click to expand full training history</summary>

```
Epoch  1/12 | Train Loss: 0.3154 | Train Acc: 84.28% | Test Acc: 92.27% | F1: 92.29% ★ New Best
Epoch  2/12 | Train Loss: 0.1993 | Train Acc: 90.69% | Test Acc: 87.84% | F1: 87.83%
Epoch  3/12 | Train Loss: 0.1522 | Train Acc: 93.66% | Test Acc: 93.56% | F1: 93.58% ★ New Best
Epoch  4/12 | Train Loss: 0.1195 | Train Acc: 94.77% | Test Acc: 97.28% | F1: 97.28% ★ New Best
Epoch  5/12 | Train Loss: 0.0848 | Train Acc: 96.26% | Test Acc: 97.21% | F1: 97.21%
Epoch  6/12 | Train Loss: 0.0686 | Train Acc: 97.47% | Test Acc: 97.71% | F1: 97.71% ★ New Best
Epoch  7/12 | Train Loss: 0.0627 | Train Acc: 97.53% | Test Acc: 96.85% | F1: 96.84%
Epoch  8/12 | Train Loss: 0.0660 | Train Acc: 97.71% | Test Acc: 97.50% | F1: 97.50%
Epoch  9/12 | Train Loss: 0.0424 | Train Acc: 98.55% | Test Acc: 98.71% | F1: 98.71% ★ New Best
Epoch 10/12 | Train Loss: 0.0466 | Train Acc: 98.28% | Test Acc: 98.21% | F1: 98.21%
Epoch 11/12 | Train Loss: 0.0370 | Train Acc: 98.39% | Test Acc: 96.35% | F1: 96.34%
Epoch 12/12 | Train Loss: 0.0375 | Train Acc: 98.71% | Test Acc: 97.07% | F1: 97.07%
```

</details>

***

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

***

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

***

## 👥 Authors

- **[________________]** — [________________]
- **[________________]** — [________________]
- **[________________]** — [________________]

***

## 🙏 Acknowledgments

- [R(2+1)D paper](https://arxiv.org/abs/1711.11248) by Tran et al. (CVPR 2018)
- [Kinetics-400](https://deepmind.com/research/open-source/kinetics) by DeepMind
- PyTorch team for pretrained video models
- Kaggle community for public fall detection datasets

***

<p align="center">
  <b>Built with ❤️ and PyTorch</b>
</p>

---

## References

1. [confusion_matrix_v2.jpg](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/153973296/10bccf11-8ace-49f2-8a78-b7230d32864a/confusion_matrix_v2.jpg?AWSAccessKeyId=ASIA2F3EMEYE62KGSXQ7&Signature=M9%2FQm9TO%2BoGzoc%2F0aGYgzzeRCXg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEAMaCXVzLWVhc3QtMSJHMEUCIElip0qjLX%2BEajuB2vsOZv71ivvt6Il4IemIEFTl8S8kAiEAv1a991OPEjaRuonvjIIIXZMmZfCAghb3cEy6NeSwuqAq%2FAQIzP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDMA1vQln2ZHOTpCZRirQBPzQE7VMG72pwMgYTqyVstLK%2FH821Zbat4Xs24ElzLmEK3SFzxvjMmMSDGCR2nhr4A1hPCW715cAAK16B3CSvskuXoH%2Bofj0sNZqrmGbUB2rwqjjUXCt2nfA%2BaJwBP6W1iIYejODN8cR2I%2FgBHneMfzner0hAeYOcEaDjaYxR8FfuBrAHU3LZMq5xCV09FwpYro4pdL8x4eeMvRH38ZU7E37ly1xz%2F6M77cWsZq%2Fhdq%2BaNVeMq0Nmo971bL5o3Mq9BqYi6ohKMsjosARvHl0AgXnjL0I40Vn0zOzM1zicpIoYOUrEguE1R2FCxLV%2BfN30GUTba%2BEk0001fiRJmX68sHXQtNE7uIise8xB2vCN4Zpcv9zc5HvyQP6PAZjtlD82yXfOR4ry9m1ahZmqsXgUQaDLioottsXC1fGgQPPVYBJ3JhT2AgSvTXeHknC98iC%2BOSvKuO8VfDrmxPT7c7yCdcdeSpE6IhyiueSB%2BOH4vPjDUxqg4pFMgakxrrINdECHtzQUGZyyc6tmBNakCJ%2FzMANYnTrvFYy3UnMrsd5FqYnIL%2BYpsJQCOg5Xea75FuEBnGZE69zZLeDn70wYihpBYtTzxxzlW4KgHl%2BfNFXTyfns2P2HRyXeTliP%2FRCseVl1nLsbtvKrqj8RjGrXpwC4VdeGMwl8vNhSHjOAzPnGmwY%2FaExOAYJM0XFaTzfHRu7tQSsKb3HMqQsSjf6RoE6d9l%2F8IlVWRN7yR8ZgkAio0feUMGxIXkabjdrbLgSokjv68JOoeXLjDE9kYnHsNwqFZww46HtzAY6mAHeElo8%2FNMfa7TlHK381hnRsIpYCKpHzvlMdoYZ0zJzrB3CUxbHbzJ%2B%2BEtTvF3NwWw4sojgtila2JbkYs0AOkBtUoYxEg2Z2bFIuVrBitXxI0c1l%2FWdqNBNb9Uk5QrtGmdowe165LOYZFLo62405UUqKu9K7bpqIdkj1l%2BcSCbDnp6odFgDzimeZwIEi7CgD8Ko1c04VMKwyg%3D%3D&Expires=1771791847)

