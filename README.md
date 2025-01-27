# Alzheimer's Disease Detection using Deep Learning

This project develops a deep learning model to classify brain images into categories associated with different stages of Alzheimer's disease. The model uses convolutional neural networks (CNN) to analyze medical images, offering a potential tool for early diagnosis of Alzheimer's disease.

---

## Problem Description

Alzheimer's disease is a progressive neurological disorder that affects memory, thinking, and behavior. Early detection can significantly improve management and care for affected individuals. The goal of this project is to build a deep learning model that can classify brain images into the following categories:

- Alzheimer's Disease (AD)
- Cognitively Normal (CN)
- Early Mild Cognitive Impairment (EMCI)
- Late Mild Cognitive Impairment (LMCI)
- Mild Cognitive Impairment (MCI)

---

## Dataset Overview

The dataset includes brain images categorized into five classes:

- **Final AD JPEG**: Alzheimer’s Disease
- **Final CN JPEG**: Cognitively Normal
- **Final EMCI JPEG**: Early Mild Cognitive Impairment
- **Final LMCI JPEG**: Late Mild Cognitive Impairment
- **Final MCI JPEG**: Mild Cognitive Impairment

### Initial Class Distribution
The dataset had an initial class imbalance:
- **Final AD JPEG**: 7,536 images
- **Final CN JPEG**: 7,430 images
- **Final EMCI JPEG**: 240 images
- **Final LMCI JPEG**: 72 images
- **Final MCI JPEG**: 922 images

To address this imbalance, data augmentation techniques were applied.

---

## Methodology

### Data Preprocessing

To balance the dataset, the following augmentation techniques were applied to classes with fewer than 1,000 images:

- **Rotation**: ±30 degrees
- **Width and Height Shifts**: Up to 20%
- **Shear**: Up to 15%
- **Zoom**: Up to 20%
- **Horizontal Flip**: Random flipping
- **Rescaling**: Normalized pixel values to the range [0, 1]

### Final Dataset Preparation
After augmentation, the dataset was split into training and testing sets:
- **Training Set**: 80% of the augmented data
- **Testing Set**: 20% of the augmented data

### Class Weight Calculation
Class weights were computed using scikit-learn's `compute_class_weight` to ensure fair training across all classes.

### Data Loading
The data was loaded using TensorFlow's `ImageDataGenerator`:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    "final_dataset/train",
    target_size=(250, 250),
    batch_size=32,
    class_mode="categorical"
)

# Load testing data
test_generator = test_datagen.flow_from_directory(
    "final_dataset/test",
    target_size=(250, 250),
    batch_size=32,
    class_mode="categorical"
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
Key Hyperparameters
Conv2D Layers: Filters of sizes 32, 64, and 128 with ReLU activation.
Pooling Layers: MaxPooling2D with pool size (2, 2).
Dense Layer: 128 units followed by a softmax output layer.
Dropout: 50% to prevent overfitting.
Optimizer: Adam optimizer.
Loss Function: Categorical Crossentropy.

Results
Training Performance
Final Training Accuracy: 95.50%
Final Validation Accuracy: 87.26%
The model performed well with consistent improvement in validation accuracy. Minor overfitting was observed, which can be mitigated using early stopping.
