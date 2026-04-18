# 🌸 Flower Classification using Deep Learning (MobileNetV2 + Streamlit)

This project is an **end-to-end deep learning system** for classifying flower images into 5 categories. It includes:

- 🧠 Model Training Notebook (`.ipynb`)
- 🌐 Interactive Web App using Streamlit

The system is built using **Transfer Learning with MobileNetV2** and deployed as a real-time image classification web application.

---

# 🚀 Project Overview

The goal is to classify flower images into the following categories:

- 🌼 Daisy  
- 🌼 Dandelion  
- 🌼 Roses  
- 🌼 Sunflowers  
- 🌼 Tulips  

---

# 🧠 Part 1: Model Training (.ipynb Notebook)

## 📂 Dataset
The model is trained on:

**TensorFlow Flowers Dataset**
- ~3,670 images
- 5 flower classes
- Train / Validation / Test split (70 / 15 / 15)

---

## ⚙️ Preprocessing

- Image resizing → 224 × 224
- Normalization using MobileNetV2 preprocessing
- Data augmentation:
  - Random Flip
  - Random Rotation
  - Random Zoom
  - Random Contrast

---

## 🏗️ Model Architecture

The model uses **:contentReference[oaicite:0]{index=0}** as a feature extractor:
Input Image (224×224×3)
      ↓
Data Augmentation
      ↓
MobileNetV2 (Pretrained on ImageNet, frozen)
      ↓
Global Average Pooling
      ↓
Dense(128, ReLU)
      ↓
Dropout(0.3)
      ↓
Dense(5, Softmax)

---

## 🔄 Training Strategy

### Phase 1: Feature Extraction
- Base model frozen
- Only classifier head trained
- Learning rate: `1e-3`

### Phase 2: Fine-Tuning
- Top layers of MobileNetV2 unfrozen
- Fine-tuned with low learning rate
- Learning rate: `1e-5`

---

## 📊 Evaluation

- Accuracy evaluated on test set
- Training vs validation accuracy/loss plots
- Prediction visualization (Actual vs Predicted)

---

## 🎯 Key Concepts Used

- Transfer Learning  
- Fine-Tuning  
- Data Augmentation  
- CNN Feature Extraction  
- Softmax Classification  

---

# 🌐 Part 2: Streamlit Web App

## 🖥️ Overview

The trained model is deployed using **Streamlit**, allowing users to interact with the model in real-time.

---

## 🚀 Features

- 📸 Upload image OR paste image URL  
- 🧠 Real-time flower classification  
- 📊 Confidence score display  
- 📈 Progress bar visualization  
- ⚡ Fast inference using trained `.h5` model  

---

## 🧩 Model Loading

The trained model is loaded using:

```python
tf.keras.models.load_model("flower_model.h5")
```
📥 Input Methods
1. Upload Image
JPG / PNG supported
Processed directly in browser
2. Image URL
Fetches image using requests
No local storage required
⚙️ Prediction Pipeline
Resize image → 224×224
Convert to NumPy array
Apply MobileNetV2 preprocessing
Expand dimensions for batch
Predict using trained model
Display:
Predicted class
Confidence score
Progress bar
📊 Output Example
🌸 Prediction: Tulips
🎯 Confidence: 92.45%
📈 Visual progress bar
🛠️ Tech Stack
Python 🐍
TensorFlow / Keras 🧠
Streamlit 🌐
NumPy 🔢
PIL (Image Processing) 🖼️
Requests 🌍
🔄 End-to-End Workflow

<img width="916" height="317" alt="image" src="https://github.com/user-attachments/assets/cdcd3341-cf90-421c-aa99-5b497ff1465c" />

▶️ Running the Application

<img width="955" height="258" alt="image" src="https://github.com/user-attachments/assets/536c4a98-e4c1-4846-821f-9381b3de88c7" />

How the application looks:
<img width="1916" height="866" alt="image" src="https://github.com/user-attachments/assets/f38181e8-0b68-4755-8f9b-3598d51d7425" />


👩‍💻 Author

Developed by Nasrin
Institute of Information Technology (IIT), Jahangirnagar University

