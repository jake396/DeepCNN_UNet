# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 17:53:28 2025

@author: Kanishk
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def down_block(x, filters, dropout_rate):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def up_block(x, skip_connection, filters):
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(filters, (2, 2), activation='relu', padding='same')(x)
    x = layers.Concatenate()([x, skip_connection])
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    return x

def build_vnet(input_shape=(256, 256, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder: DownBlocks
    d1, p1 = down_block(inputs, 256, 0.0)     # (256 → 128)
    d2, p2 = down_block(p1, 128, 0.25)        # (128 → 64)
    d3, p3 = down_block(p2, 64, 0.25)         # (64 → 32)
    d4, p4 = down_block(p3, 32, 0.25)         # (32 → 16)

    # Decoder: UpBlocks
    u1 = up_block(p4, d4, 32)                 # (16 → 32)
    u2 = up_block(u1, d3, 64)                 # (32 → 64)
    u3 = up_block(u2, d2, 128)                # (64 → 128)
    u4 = up_block(u3, d1, 256)                # (128 → 256)

    # Tail: Final output layers
    t = layers.Conv2D(2, (1, 1), activation='relu', padding='same')(u4)
    outputs = layers.Conv2D(1, (1, 1), activation='linear', padding='same')(t)

    model = models.Model(inputs, outputs, name='V-net-256')
    return model



model = build_vnet(input_shape=(256, 256, 1))
initial_learning_rate = 1e-4

train_size = 5000
batch_size = 32

steps_per_epoch = int(train_size/batch_size)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = initial_learning_rate,
    decay_steps=steps_per_epoch,
    decay_rate=1e-3,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


