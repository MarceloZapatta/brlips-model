import tensorflow as tf
from utils.data_loader import DataLoader
from model.lip_reader import LipReader
import os

# Configuration
VIDEO_DIR = "/app/data/videos"
TEXT_DIR = "/app/data/texts"
CHECKPOINT_DIR = "/app/checkpoints"
BATCH_SIZE = 32
EPOCHS = 100
MAX_SEQUENCE_LENGTH = 100

# Initialize data loader
data_loader = DataLoader(VIDEO_DIR, TEXT_DIR, MAX_SEQUENCE_LENGTH)

# Get list of all video files
video_files = [f.split('.')[0] for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]

# Load all text data first to create vocabulary
texts = []
for video_name in video_files:
    with open(os.path.join(TEXT_DIR, video_name + '.txt'), 'r') as f:
        texts.append(f.read().strip())

# Create vocabulary
vocab_size = data_loader.create_vocab(texts)

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
    filepath=os.path.join(CHECKPOINT_DIR, 'model_{epoch:02d}.h5'),
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy'
)

# Training loop
def train_generator():
    while True:
        for i in range(0, len(video_files), BATCH_SIZE):
            batch_videos = video_files[i:i+BATCH_SIZE]
            batch_x = []
            batch_y = []
            
            for video_name in batch_videos:
                frames, text = data_loader.load_sample(video_name)
                batch_x.append(frames)
                batch_y.append(data_loader.text_to_sequence(text))
                
            yield np.array(batch_x), np.array(batch_y)

# Train model
model.fit(
    train_generator(),
    steps_per_epoch=len(video_files) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback]
) 