import tensorflow as tf
from utils.data_loader import DataLoader
from model.lip_reader import LipReader, MetricsCallback
import os
import numpy as np
import time

# Configuration
VIDEO_DIR = "./app/data/videos"
TEXT_DIR = "./app/data/texts"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 1
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 100
TARGET_SPEAKER = "s6"  # Specifically use speaker s6
MAX_VIDEOS = 8  # Limit to 200 videos

# GPU Memory management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally limit GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1536)]  # Reduced to 1.5GB
        )
    except RuntimeError as e:
        print(e)

# Initialize data loader
data_loader = DataLoader(VIDEO_DIR, TEXT_DIR, MAX_SEQUENCE_LENGTH)

# Get list of videos specifically from s6
video_samples = []
speaker_dir = os.path.join(VIDEO_DIR, TARGET_SPEAKER)
if os.path.isdir(speaker_dir):
    for video_file in os.listdir(speaker_dir):
        if video_file.endswith('.mpg'):
            video_name = video_file[:-4]  # Remove .mpg extension
            video_samples.append((TARGET_SPEAKER, video_name))
else:
    print(f"Speaker directory {speaker_dir} not found!")
    exit(1)

# Limit to MAX_VIDEOS
if len(video_samples) > MAX_VIDEOS:
    np.random.shuffle(video_samples)
    video_samples = video_samples[:MAX_VIDEOS]

# Split data into training and validation sets
np.random.shuffle(video_samples)
split_idx = int(len(video_samples) * 0.8)
train_samples = video_samples[:split_idx]
val_samples = video_samples[split_idx:]

print(f"Training on {len(train_samples)} videos from {TARGET_SPEAKER}")
print(f"Validating on {len(val_samples)} videos from {TARGET_SPEAKER}")
print(f"Total videos: {len(video_samples)}")
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

# Set the idx_to_char mapping for decoding
model.set_idx_to_char(data_loader.idx_to_word)

# Compile model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    metrics=['accuracy']
)

# Create checkpoint callback
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, 'model_{epoch:02d}.weights.h5'),
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy'
)

# Add learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Create metrics callback
metrics_callback = MetricsCallback(data_loader, val_samples, model)

# Training generator
def data_generator(samples):
    while True:
        np.random.shuffle(samples)
        for i in range(0, len(samples), BATCH_SIZE):
            batch_samples = samples[i:i+BATCH_SIZE]
            print(f"\rProcessing batch {i//BATCH_SIZE + 1}/{len(samples)//BATCH_SIZE}", end="")
            batch_x = []
            batch_y = []
            input_length = []
            label_length = []
            
            # Keep track of valid samples in this batch
            valid_samples = []
            
            for speaker_id, video_name in batch_samples:
                try:
                    frames, frame_labels = data_loader.load_sample(speaker_id, video_name, augment=False)
                    
                    batch_x.append(frames)
                    # Convert labels to indices and ensure they're integers
                    sequence = np.array([data_loader.text_to_sequence(label) for label in frame_labels])
                    batch_y.append(sequence)
                    
                    # Add input and label lengths for CTC
                    input_length.append(frames.shape[0])  # Number of frames
                    label_length.append(len(sequence))
                    
                    valid_samples.append((speaker_id, video_name))
                except Exception as e:
                    print(f"\nWarning: Skipping {speaker_id}/{video_name}: {str(e)}")
                    continue
            
            # Skip this batch if no valid samples
            if not valid_samples:
                print("\nWarning: No valid samples in batch, skipping...")
                continue
                
            # Format data for CTC loss
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            input_length = np.array(input_length)
            label_length = np.array(label_length)
            
            # For CTC loss, inputs are [x, y, input_length, label_length]
            # and target is a dummy array of zeros
            inputs = {
                'input': batch_x,
                'labels': batch_y,
                'input_length': input_length,
                'label_length': label_length
            }
            outputs = np.zeros([len(batch_x), 1])  # Dummy output for CTC
            
            yield inputs, outputs

# Train model with validation split
history = model.fit(
    data_generator(train_samples),
    steps_per_epoch=len(train_samples) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=data_generator(val_samples),
    validation_steps=len(val_samples) // BATCH_SIZE,
    callbacks=[
        checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        lr_scheduler,
        metrics_callback  # Add the metrics callback
    ]
)

# Print training summary
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)

# Calculate total training time
total_epochs = len(history.history['loss'])
print(f"Total epochs trained: {total_epochs}/{EPOCHS}")

# Check if early stopping occurred
if total_epochs < EPOCHS:
    print(f"Training stopped early due to no improvement in validation loss")
    
# Find best epoch
best_epoch = np.argmin(history.history['val_loss']) + 1
best_val_loss = min(history.history['val_loss'])
best_val_acc = history.history['val_accuracy'][best_epoch-1]
print(f"Best model at epoch {best_epoch} with validation loss: {best_val_loss:.4f} and accuracy: {best_val_acc:.4f}")

# Print final metrics
final_train_acc = history.history['accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
print(f"Final training accuracy: {final_train_acc:.4f}, loss: {final_train_loss:.4f}")
print(f"Final validation accuracy: {final_val_acc:.4f}, loss: {final_val_loss:.4f}")

# Print WER and CER metrics if available
if 'real_wer' in history.history and 'real_cer' in history.history:
    final_wer = history.history['real_wer'][-1]
    final_cer = history.history['real_cer'][-1]
    print(f"Final WER: {final_wer:.4f}, CER: {final_cer:.4f}")

print("="*50)
print("Model training complete!")
print("="*50) 