# 🧠 Brain Tumor Detection with CNN

**Brain Tumor Detection with CNN** is a comprehensive deep-learning project designed to detect brain tumors from MRI images using a convolutional neural network (CNN). Developed by Jay Rathod, this project demonstrates an end-to-end pipeline—from data preprocessing to model training, evaluation, and interactive inference—offering a robust tool for exploring medical image classification and deployment.

---

## 🚀 Features

- **CNN-Based Architecture**: Utilizes a custom convolutional neural network to detect the presence (and optionally the type) of brain tumors in MRI scans.  
- **Preprocessing & Augmentation**: Includes image resizing, normalization, and augmentation to enhance model generalization on limited medical imaging data.  
- **Interactive Interface**: Built with Streamlit (or equivalent) enabling users to upload MRI images and receive a tumor or no-tumor prediction in real time.  
- **Saved Model & Tokenizer**: Ships with a pre-trained model (and optionally a serialized tokenizer/processor) so you can experiment immediately without retraining.

---

## 🧩 Project Structure
```
brain-tumor-detect-with-cnn/
│
├── dataset/ # Raw MRI image data (tumor and non-tumor classes)
├── models/ # Saved CNN model weights & checkpoints
├── app/ # Interactive web app directory
│ ├── app.py # Main user-interface script (Streamlit)
│
├── preprocess.py # Image loading, resizing, augmentation, splitting logic
├── train_model.py # Script to define, train & save the CNN model
├── infer.py # Script for loading model and predicting on new images
├── requirements.txt # Python dependencies for the project
└── README.md # Project documentation
```

## ⚙️ Installation and Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/JayRathod341997/brain-tumor-detect-with-cnn.git
cd brain-tumor-detect-with-cnn
```

### 2️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the interactive app
```bash
streamlit run app/app.py
```


🧠 Model Overview

This project employs a convolutional neural network (CNN) tailored for binary classification (tumor vs. non-tumor) of brain MRI images.

Architecture Highlights:

Input Layer: Accepts MRI image (e.g., 224×224×3 or 256×256×3) after resizing and normalization.

Convolutional + Pooling Blocks: Stacked convolutional filters extract hierarchical features; max-pooling reduces spatial dimensions.

Dropout / Batch Normalization: Used to reduce overfitting and stabilize training.

Dense Layers: Fully connected layers culminating in a sigmoid (for binary) or softmax (for multi-class) activation.

Output Layer: Predicts the probability of a brain tumor being present (or class label).
