import json
import os
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='/content/drive/MyDrive/Model_checkpoints/best_model.h5',  # Where to save the model
    monitor='val_loss',        # What metric to track
    save_best_only=True,       # Only save if val_loss improves
    mode='min',                # Because lower val_loss is better
    verbose=1
)

class SaveHistoryJSON(Callback):
    def __init__(self, filepath='training_history.json'):
        super().__init__()
        self.filepath = filepath
        self.history_data = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history_data.setdefault(key, []).append(float(value))

        # Save after each epoch
        with open(self.filepath, 'w') as f:
            json.dump(self.history_data, f)

    def on_train_end(self, logs=None):
        # Final save (optional redundancy)
        with open(self.filepath, 'w') as f:
            json.dump(self.history_data, f)


history = model.fit(
    train_dataset,
    steps_per_epoch=train_size // batch_size,
    validation_data=val_dataset,
    validation_steps=val_size // batch_size,
    epochs=50,
    callbacks=[checkpoint, history_callback]
)


from tensorflow.keras.models import load_model

# Load without compiling to avoid errors
model = load_model('/content/drive/MyDrive/Model_checkpoints/best_model.h5', compile=False)

# Then recompile with original settings
model.compile(optimizer='adam', loss='mae', metrics=['mae'])  # or your actual loss/metrics