import tensorflow as tf
from utils.data_loader import DataLoader
from model.lip_reader import LipReader
import os
import numpy as np

# Configuration
VIDEO_DIR = "./app/data/videos"
TEXT_DIR = "./app/data/texts"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 2  # Small batch size for testing
EPOCHS = 5      # Few epochs for testing
MAX_SEQUENCE_LENGTH = 100
# Add memory management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Initialize data loader
data_loader = DataLoader(VIDEO_DIR, TEXT_DIR, MAX_SEQUENCE_LENGTH)

# Get just two videos for testing
test_samples = []
for speaker_id in os.listdir(VIDEO_DIR):
    speaker_dir = os.path.join(VIDEO_DIR, speaker_id)
    if os.path.isdir(speaker_dir):
        video_files = [f for f in os.listdir(speaker_dir) if f.endswith('.mpg')][:2]  # Get first 2 videos
        for video_file in video_files:
            video_name = video_file[:-4]  # Remove .mpg extension
            test_samples.append((speaker_id, video_name))
            if len(test_samples) >= 2:  # Stop after getting 2 videos
                break
    if len(test_samples) >= 2:  # Stop after getting 2 videos
        break

print("Testing with videos:", test_samples)

# Load text data for test samples
texts = []
for speaker_id, video_name in test_samples:
    text_path = os.path.join(TEXT_DIR, speaker_id, f"{video_name}.align")
    with open(text_path, 'r') as f:
        text = f.read().strip()
        texts.append(text)
        print(f"Text for {speaker_id}/{video_name}: {text[:50]}...")  # Show first 50 chars

# Create vocabulary from test samples
vocab_size = data_loader.create_vocab(texts)
print(f"Test vocabulary size: {vocab_size}")

# Create model
model = LipReader(vocab_size, MAX_SEQUENCE_LENGTH)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Create checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, 'test_model_{epoch:02d}.weights.h5'),
    save_weights_only=True,
    save_best_only=True,
    monitor='loss'  # Use loss instead of val_accuracy since we're not using validation
)

# Training generator for test samples
def test_train_generator():
    while True:
        batch_x = []
        batch_y = []
        
        for speaker_id, video_name in test_samples:
            print(f"\nProcessing {speaker_id}/{video_name}")
            frames, text = data_loader.load_sample(speaker_id, video_name)
            batch_x.append(frames)
            batch_y.append(data_loader.text_to_sequence(text, pad_to_length=MAX_SEQUENCE_LENGTH))
            
        yield np.array(batch_x), np.array(batch_y)

# Train model with test samples
print("\nStarting test training...")
model.fit(
    test_train_generator(),
    steps_per_epoch=1,  # One step per epoch since we're using all samples in one batch
    epochs=EPOCHS,
    callbacks=[checkpoint_callback]
)

print("\nTest training completed!") 