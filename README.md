# Fall Detection AI 🚀

**3D CNN trained on 6,988 videos** | **Test Accuracy: 71.03% F1: 71.09%**

[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-71.03%25-brightgreen)](https://github.com/Mortezamohasebati/Folio_Finder_AI/blob/main/confusion_matrix.png)
[![Model Size](https://img.shields.io/badge/Model-196MB-orange)](https://github.com/Mortezamohasebati/Folio_Finder_AI/releases)

## 🎯 Features
- **Trained on**: 6,988 fall/no-fall videos
- **Model**: Simple3D CNN (PyTorch)
-**Training Peak**: **91.8%**
- **Test Results**: 71% accuracy, balanced F1-score
- **Live prediction**: Any MP4 video

## 🚀 Quick Start

```bash
# Clone + install
git clone https://github.com/Mortezamohasebati/Folio_Finder_AI.git
cd Folio_Finder_AI
pip install -r requirements.txt

# Predict fall on YOUR video (model auto-downloads via LFS)
python predict_fall.py your_video.mp4


