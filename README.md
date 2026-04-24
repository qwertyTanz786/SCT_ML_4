# 🖐️ Hand Gesture Recognition using CNN

A Deep Learning-based Hand Gesture Recognition System developed using Convolutional Neural Networks (CNN) and the LeapGestRecog dataset for intuitive Human-Computer Interaction (HCI). This project classifies different hand gestures from image data with high accuracy and demonstrates the practical use of Computer Vision in gesture-based systems.

---

## 📌 Project Overview

Hand Gesture Recognition is an important application of Computer Vision and Deep Learning that enables machines to understand human hand movements and gestures.

This project uses the LeapGestRecog dataset containing 20,000+ hand gesture images across 10 different gesture classes. A CNN model is trained to accurately classify these gestures and enable gesture-based interaction systems.

The system performs:

- Image preprocessing  
- Gesture classification using CNN  
- Model training and testing  
- High-accuracy prediction  
- Deployment-ready model saving  

---

## 🎯 Objective

To develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

---

## 📂 Dataset Used

Dataset: LeapGestRecog Dataset

- 20,000+ grayscale hand gesture images  
- 10 gesture classes  
- 10 different subjects  
- Captured using Leap Motion Controller  
- Infrared image dataset for better consistency  

Dataset Link:  
https://www.kaggle.com/datasets/gti-upm/leapgestrecog

---

## ✋ Gesture Classes

The model classifies the following 10 gestures:

- Palm  
- L  
- Fist  
- Fist Moved  
- Thumb  
- Index  
- OK  
- Palm Moved  
- C  
- Down  

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- NumPy  
- TensorFlow / Keras  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) consisting of:

- Conv2D Layer (32 filters)  
- MaxPooling Layer  
- Conv2D Layer (64 filters)  
- MaxPooling Layer  
- Conv2D Layer (128 filters)  
- MaxPooling Layer  
- Flatten Layer  
- Dense Layer (128 neurons)  
- Dropout Layer  
- Output Layer (10 classes with Softmax)  

This architecture helps the model learn important gesture features effectively.

---
# 📊 Hand Gesture Recognition Model Output

This section shows the final output results and performance metrics of the CNN-based Hand Gesture Recognition System trained on the LeapGestRecog dataset.
---

## ✅ Final Output
Loading images...
Dataset Loaded Successfully!

Training Data Shape: (16000, 64, 64, 1)
Testing Data Shape: (4000, 64, 64, 1)

Epoch 1/15
...
Epoch 15/15
Training Complete

Test Accuracy: 100.00%
Test Loss: 0.0003

Model Saved Successfully!
