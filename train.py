import tensorflow as tf
from utils.data_loader import DataLoader
from model.lip_reader import LipReader
import os
import numpy as np

# Configuration
VIDEO_DIR = "./app/data/videos"
TEXT_DIR = "./app/data/texts"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 4  # Reduced batch size
EPOCHS = 100
MAX_SEQUENCE_LENGTH = 100

# GPU Memory management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally limit GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]  # Limit to 2GB
        )
    except RuntimeError as e:
        print(e)

# Initialize data loader
data_loader = DataLoader(VIDEO_DIR, TEXT_DIR, MAX_SEQUENCE_LENGTH)

# Get list of all speakers and their videos
video_samples = []
for speaker_id in os.listdir(VIDEO_DIR):
    speaker_dir = os.path.join(VIDEO_DIR, speaker_id)
    if os.path.isdir(speaker_dir):
        for video_file in os.listdir(speaker_dir):
            if video_file.endswith('.mpg'):
                video_name = video_file[:-4]  # Remove .mpg extension
                video_samples.append((speaker_id, video_name))

# Split data into train and validation
np.random.shuffle(video_samples)
split_idx = int(len(video_samples) * 0.8)  # 80% for training
train_samples = video_samples[:split_idx]
val_samples = video_samples[split_idx:]
print(f"Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

print(f"Found {len(video_samples)} video files")
print("First few samples:", video_samples[:5])

# Load all text data first to create vocabulary
texts = []
for speaker_id, video_name in video_samples:
    text_path = os.path.join(TEXT_DIR, speaker_id, f"{video_name}.align")
    with open(text_path, 'r') as f:
        texts.append(f.read().strip())

# Create vocabulary
vocab_size = data_loader.create_vocab(texts)
print(f"Vocabulary size: {vocab_size}")
print("Sample text:", texts[0][:100])

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
    filepath=os.path.join(CHECKPOINT_DIR, 'model_{epoch:02d}.weights.h5'),
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy'
)

# Training generator
def data_generator(samples):
    while True:
        np.random.shuffle(samples)
        for i in range(0, len(samples), BATCH_SIZE):
            batch_samples = samples[i:i+BATCH_SIZE]
            print(f"\rProcessing batch {i//BATCH_SIZE + 1}/{len(video_samples)//BATCH_SIZE}", end="")
            batch_x = []
            batch_y = []
            
            for speaker_id, video_name in batch_samples:
                try:
                    frames, frame_labels = data_loader.load_sample(speaker_id, video_name)
                    batch_x.append(frames)
                    # Convert labels to indices and ensure they're integers
                    sequence = np.array([data_loader.text_to_sequence(label) for label in frame_labels])
                    batch_y.append(sequence)
                except Exception as e:
                    print(f"\nError processing {speaker_id}/{video_name}:")
                    print(f"Video path: {os.path.join(VIDEO_DIR, speaker_id, f'{video_name}.mpg')}")
                    print(f"Align path: {os.path.join(TEXT_DIR, speaker_id, f'{video_name}.align')}")
                    print(f"Error: {str(e)}")
                    raise  # Re-raise the exception to stop training
                
            # Convert to numpy arrays with correct shapes
            batch_x = np.array(batch_x)  # Shape: (batch_size, max_seq_len, height, width)
            batch_y = np.array(batch_y)  # Shape: (batch_size, max_seq_len)
            yield batch_x, batch_y

# Train model with validation split
model.fit(
    data_generator(train_samples),
    steps_per_epoch=len(video_samples) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=data_generator(val_samples),
    validation_steps=len(val_samples) // BATCH_SIZE,
    callbacks=[
        checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        )
    ]
) 