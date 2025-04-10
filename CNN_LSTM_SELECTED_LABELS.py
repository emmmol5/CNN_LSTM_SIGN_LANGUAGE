from IMPORTS import *

from tensorflow.keras.applications import MobileNetV2
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import random
import matplotlib.pyplot as plt

# Paths and constants
zero_pad_resized = r"C:\Users\Emma\OneDrive - Noroff Education AS\3. Året\Bachelor\Sign Language Health\vid_zeropad_resized"
IMG_SIZE = 128
NUM_FRAMES = 80
BATCH_SIZE = 16
EPOCHS = 100

# Load and normalize video
def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) < NUM_FRAMES:
        return None

    frames = np.array(frames[:NUM_FRAMES]).astype("float32") / 255.0
    return frames

# Data generator that skips broken/short videos
def data_generator(video_paths, labels, batch_size=BATCH_SIZE):
    while True:
        idxs = list(range(len(video_paths)))
        random.shuffle(idxs)

        batch_videos, batch_labels = [], []

        for idx in idxs:
            video = load_video(video_paths[idx])
            if video is not None:
                batch_videos.append(video)
                batch_labels.append(labels[idx])
            if len(batch_videos) == batch_size:
                yield np.array(batch_videos), np.array(batch_labels)
                batch_videos, batch_labels = [], []

# Build model
def build_model(input_shape, num_classes):
    base_cnn = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-30]:
        layer.trainable = False

    model_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.TimeDistributed(base_cnn)(model_input)
    x = tf.keras.layers.LSTM(128, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=model_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset with only a few random classes
all_labels = [label for label in os.listdir(zero_pad_resized) if os.path.isdir(os.path.join(zero_pad_resized, label))]
selected_labels = random.sample(all_labels, 10)  # use 10 random classes

train_paths, val_paths, train_labels, val_labels = [], [], [], []

for label in selected_labels:
    label_path = os.path.join(zero_pad_resized, label)
    videos = [f for f in os.listdir(label_path) if f.endswith(".mp4")]
    random.shuffle(videos)

    full_paths = [os.path.join(label_path, f) for f in videos]

    train_paths.extend(full_paths[:8])
    train_labels.extend([label] * 8)
    val_paths.extend(full_paths[8:10])
    val_labels.extend([label] * 2)

# Encode labels based only on selected classes
NUM_CLASSES = len(selected_labels)
label_encoder = LabelEncoder()
label_encoder.fit(selected_labels)

train_encoded = to_categorical(label_encoder.transform(train_labels), num_classes=NUM_CLASSES)
val_encoded = to_categorical(label_encoder.transform(val_labels), num_classes=NUM_CLASSES)

# Generators
train_gen = data_generator(train_paths, train_encoded)
val_gen = data_generator(val_paths, val_encoded)

steps_per_epoch = max(1, len(train_paths) // BATCH_SIZE)
val_steps = max(1, len(val_paths) // BATCH_SIZE)

# Build model
input_shape = (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
model = build_model(input_shape, NUM_CLASSES)
model.summary()

# Optional callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("cnn_lstm_best_model.keras", save_best_only=True, monitor="val_accuracy")
]

# Train model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
test_loss, test_acc = model.evaluate(val_gen, steps=val_steps)
print(f"Final Accuracy: {test_acc:.4f}")

# Plot accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
