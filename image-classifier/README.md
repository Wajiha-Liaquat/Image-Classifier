# 🧠 CIFAR-10 Image Classifier (PyTorch)

This project implements a **Convolutional Neural Network (CNN)** for image classification on a CIFAR-10–style dataset using **PyTorch**. It includes data preprocessing, custom dataset handling, model training, evaluation, and visualization of results.

---

## 📁 Project Structure

image-classifier/
│
├── main.py                  # Entry point for training pipeline
├── requirements.txt        # Dependencies
│
├── dataset/
│   └── custom_dataset.py   # Dataset creation & preprocessing
│
├── models/
│   └── model.py            # CNN architecture
│
├── train/
│   └── train.py            # Training loop
│
├── utils/
│   └── helper.py           # Evaluation & visualization
│
├── plots/
│   ├── trial_1_summary.png
│   ├── trial_2_summary.png
│   └── trial_3_summary.png
│
└── results/
    └── best_model.pth      # Saved best model

---

## 🚀 Features

- Custom **PyTorch Dataset Class**
- Automated **data organization & train-test split**
- CNN with:
  - Batch Normalization
  - Max Pooling
- Multiple training configurations (trials)
- Performance tracking:
  - Loss
  - Accuracy
- Visualization of training metrics
- Best model checkpoint saving

---

## 🧾 Dataset Preparation

The dataset is expected in the following format:

### Step 1: Raw Dataset
- Images in a folder (e.g., `/train`)
- Labels in CSV file:

id,label
1,cat
2,dog
...

### Step 2: Organize Data

Use dataset utilities to:
- Create class-wise folders
- Split into train and test sets

---

## ⚙️ Installation

Install dependencies:

pip install -r requirements.txt

---

## ▶️ How to Run

python main.py

---

## 🧪 Training Configuration

The pipeline runs 3 experiments:

| Trial | Epochs | Optimizer | Learning Rate | Batch Size |
|------|--------|----------|--------------|-----------|
| 1    | 5      | SGD      | 0.01         | 32        |
| 2    | 10     | Adam     | 0.001        | 64        |
| 3    | 20     | Adam     | 0.001        | 64        |

---

## 🧠 Model Architecture

- 3 Convolutional Layers:
  - 3 → 32 → 64 → 128 channels
- Batch Normalization after each conv layer
- Max Pooling
- Fully Connected Layers:
  - 2048 → 256 → 10

---

## 📊 Outputs

- Training logs (loss & accuracy per epoch)
- Saved plots:
  plots/trial_X_summary.png
- Best model:
  results/best_model.pth

---

## 📈 Evaluation

Evaluation includes:
- CrossEntropy Loss
- Accuracy (%)

---

## 🛠️ Key Components

### 🔹 Custom Dataset
Handles:
- Folder-based class mapping
- Image loading
- Transformations

### 🔹 Training Loop
- Supports SGD & Adam
- Tracks metrics per epoch

### 🔹 Visualization
- Loss curve
- Accuracy curve

---

## 💡 Notes

- Designed for Google Colab paths (`/content/...`)
- Modify dataset paths if running locally
- GPU automatically used if available

---

## 📌 Future Improvements

- Add validation split
- Early stopping
- Data augmentation
- Hyperparameter tuning
- Transfer learning (ResNet, EfficientNet)

---

## 👤 Author

**Wajiha (MSCS25016)**
