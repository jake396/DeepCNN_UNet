import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

# === STEP 3: Define Your CNN Model ===
model = models.Sequential()

# Input layer
model.add(layers.Input(shape=(50, 50, 1)))

# Layers 1–5: 8 filters
for _ in range(5):
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))

# Layers 6–8: 32 filters
for _ in range(3):
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))

# Layers 9–11: 64 filters
for _ in range(3):
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

# Layers 12–14: 32 filters
for _ in range(3):
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))

# Layers 15–19: 8 filters
for _ in range(5):
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))

# Final output layer: 1×1 conv, 1 filter, no activation
model.add(layers.Conv2D(1, (1, 1), activation=None, padding='same'))