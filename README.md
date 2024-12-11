# Pawers Machine Learning

Pawers is an application designed to detect cat skin diseases through computer vision and a chatbot model.

## Project Overview

The objective of this project is to detect and diagnose cat skin diseases using a CNN model and a chatbot model. With this combination, the application provides visual predictions and text-based interactions to assist users.

## Dataset

The dataset consists of a collection of cat skin disease images obtained through web scraping and is used to train the CNN model.

- [Dataset Link](https://drive.google.com/drive/folders/1cEPHoN7FJ-5H1LAwhcKVveqzQxv1hIJE?usp=sharing)

## Features

### CNN Model:

- Data Augmentation
- CNN (Convolutional Neural Networks)
- Transfer Learning (MobileNetV2)

### Chatbot Model:

- Tokenization using NLTK
- Intent classification with a neural network-based model
- Text preprocessing, including stemming and stop-word removal
- Responses based on an intents JSON dataset

## Requirements

- TensorFlow
- Keras
- Matplotlib
- NumPy
- Pillow
- NLTK
- Torch
- Flask

## Documentation

1. **CNN Model:**
   - The dataset was processed with data augmentation to increase training data diversity.
   - The model was built using the MobileNetV2 architecture with transfer learning for efficiency.
   - The model was trained to detect various classes of cat skin diseases.
2. **Chatbot Model:**
   - The intents dataset is in JSON format, containing a list of intents and related responses.
   - Text preprocessing ensures high accuracy in intent classification.
   - A neural network model was trained to recognize user intents and provide appropriate responses.

## Future Work

- Expand the dataset to include more classes of cat skin diseases.
- Explore advanced architectures and methodologies to further enhance accuracy.