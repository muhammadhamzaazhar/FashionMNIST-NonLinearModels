# Modeling Non-Linear Data_Enhancing Accuracy with Multiple Approaches
## Overview

Non-linear classification problems often require more sophisticated models than linear classifiers. This notebook explores how different neural network designs and hyperparameters (activation functions, learning rates, scheduling) can improve accuracy on a synthetic circular dataset.

## Notebook Contents

1. **Generate Synthetic Dataset**  
   Create a non-linearly separable dataset (concentric circles) using `sklearn.datasets.make_circles`.

2. **Data Visualization**  
   Plot the dataset to illustrate the classification challenge.

3. **Model Architectures**  
   Implement and train six distinct neural networks, varying:
   - Number of layers and units
   - Activation functions (ReLU, Sigmoid)
   - Learning rate schedules

4. **Training & Evaluation**  
   Track loss curves, compute confusion matrices, and compare model accuracy.

5. **Conclusion**  
   Summarize findings and best practices for non-linear classification.

---

# Multi-Class Classification of Fashion MNIST
This repository contains a Jupyter Notebook titled **`Multi-Class Classification of Fashion MNIST.ipynb`**, which explores the Fashion MNIST dataset. The notebook includes data loading, visualization of class labels, and prepares the dataset for a multi-class image classification task.

## Dataset Overview

**Fashion MNIST** is a dataset of **70,000 grayscale images** of size **28x28 pixels**, distributed as follows:

- **60,000 training images**
- **10,000 test images**
- **10 clothing categories**, each labeled from `0` to `9`:

| Label | Class        |
|-------|--------------|
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle boot   |

> Source: [Fashion MNIST – Zalando Research](https://github.com/zalandoresearch/fashion-mnist)


## Visualizations

The notebook includes the following visualizations to explore the dataset:

- **Single Image Display**  
  View an individual training image along with its label (e.g., index `3` shows "Dress").

- **Grid of Samples**  
  Display a `5x5` grid of 25 images from the training set with their corresponding class names to gain an intuitive understanding of the dataset.

## Models Used

This project implements and compares multiple neural network models for Fashion MNIST classification:

- **Model 1: Shallow Fully Connected Neural Network (Without Normalization)**  
  A basic MLP trained directly on raw pixel values.

- **Model 2: Shallow Fully Connected Neural Network (With Normalization)**  
  Same as Model 1 but with input normalization to improve training dynamics.

- **Model 3: Deep Fully Connected Neural Network (Learning Rate Finder)**  
  A deeper MLP architecture used to experiment with learning rate scheduling and optimization.

- **Model 4: Tuned Deep Fully Connected Neural Network (Optimal Learning Rate)**  
  A fine-tuned version of Model 3 trained with the optimal learning rate for enhanced accuracy and convergence.

> All models use fully connected (dense) layers. Convolutional architectures are not used in this version.


## Evaluation

- **Confusion Matrix Visualization**  
  Used to evaluate class-wise performance after model prediction on the test set.

- **Random Image Prediction Plotting**  
  Displays a set of test images alongside their predicted and actual labels for intuitive model performance understanding.
---

