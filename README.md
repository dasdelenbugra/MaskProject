# Mask Detection with CNN

A deep learning project that classifies whether a person is wearing a face mask using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. Trained on Google Colab with GPU acceleration.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dasdelenbugra/MaskProject/blob/main/MaskProject.ipynb)

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **90.91%** |
| Precision (mask) | 92.3% |
| Recall (mask) | 87.8% |
| Dataset size | 440 images |
| Epochs | 10 |

### Confusion Matrix

|  | Predicted: No Mask | Predicted: Mask |
|--|--|--|
| **Actual: No Mask** | 44 ✅ | 3 ❌ |
| **Actual: Mask** | 5 ❌ | 36 ✅ |

---

## How It Works

1. Images are resized to **128×128** pixels and normalized (pixel values 0–1)
2. A CNN model is trained to classify images into two classes: `with_mask` / `without_mask`
3. At inference time, a single image path is provided and the model predicts the class

---

## Model Architecture

```
Input (128 × 128 × 3)
  → Conv2D(32, 3×3, ReLU)
  → MaxPooling2D(2×2)
  → Conv2D(64, 3×3, ReLU)
  → MaxPooling2D(2×2)
  → Flatten
  → Dense(128, ReLU)
  → Dropout(0.5)
  → Dense(64, ReLU)
  → Dropout(0.5)
  → Dense(2, Softmax)

Optimizer : Adam
Loss      : Sparse Categorical Crossentropy
```

---

## Dataset

- **440 images** total — 220 with mask, 220 without mask
- **Train / Test split:** 80% / 20% (352 train, 88 test)
- Images resized to 128×128 RGB, pixel values normalized to [0, 1]

---

## Technologies

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| TensorFlow / Keras | Model building & training |
| OpenCV | Image reading & processing |
| NumPy | Array operations |
| Scikit-learn | Train/test split, confusion matrix |
| Matplotlib | Training curves visualization |
| Google Colab | GPU-accelerated training environment |

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/dasdelenbugra/MaskProject.git
cd MaskProject

# Install dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn pillow

# Open the notebook
jupyter notebook MaskProject.ipynb
```

### Predict a single image

```python
# At the end of the notebook, enter the image path when prompted:
Path of the image to be predicted: /path/to/your/image.jpg

# Output example:
# Görüntüdeki kişi maske takmıştır.  →  Person IS wearing a mask
# Görüntüdeki kişi maske takmamıştır. →  Person is NOT wearing a mask
```

---

## Project Structure

```
MaskProject/
├── MaskProject.ipynb    # Full training & evaluation notebook
└── README.md
```

---

## Author

**Ömer Buğra Daşdelen**  
Computer Engineering Graduate | AI/ML & Android Developer  
[GitHub](https://github.com/dasdelenbugra) · [LinkedIn](https://linkedin.com/in/ömer-buğra-daşdelen-24746124b)

