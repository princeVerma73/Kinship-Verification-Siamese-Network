# 👨‍👩‍👧 Kinship Verification using Siamese Neural Network

A Deep Learning project that predicts whether two face images belong to biological family members using a Siamese Neural Network built with PyTorch and deployed with Streamlit.

---

## 📌 Overview

Kinship verification is a binary classification task where the model determines whether two face images belong to related people.

This project explores multiple training strategies:

- Baseline Siamese Network
- Train-Faces Dataset
- Combined Training Strategy

The final application provides an interactive Streamlit interface for uploading two face images and obtaining the prediction instantly.

---

## 🚀 Features

- Siamese Neural Network using ResNet18 backbone
- Face pair similarity prediction
- Streamlit Web Application
- GPU/CPU Support
- PyTorch Inference Pipeline
- Multiple Training Experiments
- Clean Project Structure

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- PIL (Pillow)
- NumPy
- Kaggle Dataset

---

## 📂 Project Structure

```
Kinship-Verification-Siamese-Network/
│
├── app.py                      # Streamlit application
├── model.py                    # Siamese Network architecture
├── predict.py                  # Prediction pipeline
├── utils.py                    # Image preprocessing
│
├── 01_train_baseline.ipynb     # Baseline training
├── 02_train_faces.ipynb        # Training on train-faces dataset
├── 03_combined_training.ipynb  # Combined training
│
├── checkpoints_baseline/
├── checkpoints_train_faces/
├── checkpoints_combined/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/princeVerma73/Kinship-Verification-Siamese-Network.git
```

Go inside the project

```bash
cd Kinship-Verification-Siamese-Network
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit App

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

## 🧠 Model Workflow

```
Face Image 1
        │
        ▼
    ResNet18
        │
   Face Embedding
        │
        │
Face Image 2
        │
        ▼
    ResNet18
        │
   Face Embedding
        │
        ▼
Embedding Comparison
        │
        ▼
Related / Not Related
```

---

## 📷 Application Preview

Upload two face images.

The application will:

- Preprocess both images
- Generate embeddings
- Predict whether the pair belongs to related family members
- Display confidence score

---

## 📊 Training Experiments

| Notebook | Purpose |
|----------|---------|
| 01_train_baseline.ipynb | Baseline Siamese Network |
| 02_train_faces.ipynb | Training using train-faces dataset |
| 03_combined_training.ipynb | Combined training strategy |

---

## 📌 Future Improvements

- Better face alignment
- ArcFace embeddings
- EfficientNet backbone
- Ensemble models
- Improved threshold calibration
- Docker deployment
- Cloud deployment

---

## 👨‍💻 Author

**Prince Verma**

B.Tech CSE  
IIIT Bhagalpur

GitHub

https://github.com/princeVerma73

LinkedIn

https://www.linkedin.com/in/princeverma73/

---

## ⭐ If you like this project

Give the repository a ⭐ on GitHub.
