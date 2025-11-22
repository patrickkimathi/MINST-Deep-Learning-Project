# MNIST Handwritten Digit Classification using Artificial Neural Networks (ANN)

## Student: Patrick Kimathi Kariuki

## Course: Cyber Shujaa -- Data Science

## Assignment 10: Deep Learning

## Project Overview

This project applies deep learning techniques using Artificial Neural
Networks (ANNs) built with TensorFlow/Keras to classify handwritten
digits from the MNIST dataset. MNIST is a benchmark dataset widely used
for testing image classification models and consists of 70,000 grayscale
images representing digits from 0 to 9. The goal of this assignment is
to demonstrate understanding of building, training, evaluating, and
saving a deep learning model for image recognition.

## Dataset Summary (MNIST)

-   Images: 70,000 grayscale digit images\
-   Resolution: 28×28 pixels\
-   Classes: 10 (digits 0--9)\
-   Split: 60,000 training images, 10,000 test images

## Objectives

-   Load and preprocess image datasets\
-   Visualize image samples\
-   Design a deep learning ANN using Sequential API\
-   Train, validate, and evaluate model performance\
-   Generate predictions and confusion matrix\
-   Report accuracy, precision, recall, and F1-score\
-   Save and reload trained models using the Keras native format

## Technologies Used

-   Python\
-   NumPy\
-   Matplotlib / Seaborn\
-   TensorFlow / Keras\
-   scikit-learn

## Project Workflow

### 1. Data Loading & Preprocessing

-   Loaded MNIST dataset via tensorflow.keras.datasets
-   Normalized pixel values to the range \[0, 1\]
-   One-hot encoded the labels
-   Displayed 9 sample images

### 2. Model Architecture (Sequential Model)

-   Flatten Layer\
-   Dense (128 neurons, ReLU)\
-   Dropout (0.3)\
-   Dense (64 neurons, ReLU)\
-   Dropout (0.3)\
-   Output Layer (10 neurons, Softmax)

### 3. Model Compilation

-   Optimizer: Adam\
-   Loss: categorical_crossentropy\
-   Metric: accuracy

### 4. Model Training

-   10 epochs\
-   Batch size: 128\
-   Validation split: 0.1

### 5. Model Evaluation

-   Confusion matrix\
-   Classification report

### 6. Model Saving

-   Saved using: model.save('mnist_model.h5')

## Results Summary

-   Achieved high accuracy (typically above 97%)\
-   Balanced precision and recall across classes

## Files Included

-   Python script\
-   Saved model (.h5)\
-   README.md

## How to Run

1.  Clone the repository\
2.  Install dependencies\
3.  Run the script

## Acknowledgment

This project was completed as part of the Cyber Shujaa Data Science
Program.
