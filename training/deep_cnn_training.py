def custom_lr_schedule(epoch):
    if epoch < 7:
        return 1e-3
    elif epoch < 13:
        return 6e-4
    elif epoch < 19:
        return 3.6e-4
    elif epoch < 25:
        return 2.16e-4
    elif epoch < 31:
        return 1.296e-4
    elif epoch < 37:
        return 0.7776e-4
    elif epoch < 43:
        return 0.4666e-4
    elif epoch < 49:
        return 0.2799e-4
    elif epoch < 55:
        return 0.168e-4
    else:
        return 0.1008e-4

lr_callback = tf.keras.callbacks.LearningRateScheduler(custom_lr_schedule)

# === STEP 5: Compile and Train the Model ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mean_absolute_error',
    metrics=['mae']
)

history = model.fit(
    train_noisy, train_clean,
    epochs=60,
    batch_size=32,
    verbose=2,
    callbacks=[lr_callback]
)