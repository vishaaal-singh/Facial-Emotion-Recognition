# Facial Emotion Recognition - Multi Classification

## Overview
This project aims to classify facial emotions using multi-image classification techniques. The model leverages a Convolutional Neural Network (CNN) to recognize and categorize facial emotions such as happy, sad, angry, and neutral. The dataset is loaded from a directory of labeled images, and the model is trained using TensorFlow and Keras.

## Dataset Information
The objective of the project is to detect facial expression using image dataset. Convolutional Neural Network is used to classify the images. The output class consists of 7 different types namely angry, disgust, fear, happy, neutral, sad, surprise.

Download link: https://www.kaggle.com/aadityasinghal/facial-expression-dataset

Environment: kaggle

## Key Features
- **Multi-Class Image Classification**: Identifies emotions from facial images in different classes.
- **Deep Learning**: Built using a Convolutional Neural Network (CNN) for efficient feature extraction.
- **Data Augmentation**: Uses augmentation techniques to improve model generalization and performance.
- **Real-Time Prediction**: Optional functionality to detect facial expressions from a webcam or video feed.

## Dataset
The dataset consists of labeled images of human faces, each associated with one of the following emotion categories:
- Happy
- Sad
- Angry
- Neutral

The dataset is stored in separate directories for training and testing, with further subdirectories representing each class.

### Dataset Structure
- Training Set: Contains images for training the model.
- Test Set: Contains images for evaluating the model's performance.

## Project Workflow

1. **Data Preprocessing**:
   - Resizes images and converts them to a format suitable for training.
   - Applies augmentation techniques (rotation, zoom, horizontal flip) to improve the model’s robustness.
   
2. **Model Training**:
   - Utilizes TensorFlow and Keras to build a CNN architecture for emotion classification.
   - The model includes layers like Conv2D, MaxPooling2D, Flatten, and Dense for classification.
   
3. **Model Evaluation**:
   - After training, the model is evaluated on the test dataset to assess its performance.
   - Key metrics such as accuracy, loss, and confusion matrix are used to measure success.
   
4. **Real-Time Emotion Detection**:
   - The model can optionally be extended to classify emotions in real-time using a webcam.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Seaborn

