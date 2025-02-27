from utils.video_processor import VideoProcessor
from utils.data_loader import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from model.lip_reader import LipReader

# Configuration
VIDEO_DIR = "./app/data/videos"
TEXT_DIR = "./app/data/texts"
MAX_SEQUENCE_LENGTH = 100
OUTPUT_DIR = "./word_visualizations"
CHECKPOINT_DIR = "./checkpoints"
TARGET_SPEAKER = "s6"  # Use the same speaker as in training

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize data loader and processor
data_loader = DataLoader(VIDEO_DIR, TEXT_DIR, MAX_SEQUENCE_LENGTH)
processor = VideoProcessor()

def generate_word_examples(speaker_id, video_name):
    """Generate mouth image examples for different words"""
    print(f"Processing {speaker_id}/{video_name}")
    
    try:
        # Load and process video
        frames, frame_labels = data_loader.load_sample(speaker_id, video_name)
        
        # Get unique words (excluding silence)
        unique_words = []
        for word in frame_labels:
            if word != 'sil' and word not in unique_words:
                unique_words.append(word)
        
        # Limit to 5 words
        if len(unique_words) > 5:
            # Try to select interesting words (not just common ones)
            word_counts = {}
            for word in frame_labels:
                if word != 'sil':
                    word_counts[word] = word_counts.get(word, 0) + 1
                    
            # Sort by frequency (less frequent first to get more interesting words)
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1])
            selected_words = [word for word, _ in sorted_words[:5]]
            unique_words = selected_words
        
        print(f"Selected {len(unique_words)} words: {', '.join(unique_words)}")
        
        # For each unique word, find frames where it appears
        for word in unique_words:
            word_frames = []
            frame_indices = []
            
            # Find all frames with this word
            for i, label in enumerate(frame_labels):
                if label == word:
                    word_frames.append(frames[i])
                    frame_indices.append(i)
            
            # Skip if no frames found
            if not word_frames:
                continue
                
            # Save individual frames with word in filename
            for idx, (frame, frame_idx) in enumerate(zip(word_frames[:3], frame_indices[:3])):  # Limit to 3 frames per word
                # Create a clean filename (remove special characters)
                clean_word = ''.join(c if c.isalnum() else '_' for c in word)
                output_path = os.path.join(
                    OUTPUT_DIR, 
                    f"{speaker_id}_{video_name}_{clean_word}_frame{idx}.png"
                )
                
                # Save the raw mouth image
                plt.figure(figsize=(4, 4))
                plt.imshow(frame, cmap='gray')
                plt.title(f"Word: '{word}'")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                print(f"Saved {word} example to {output_path}")
            
            # Also create a grid of all frames for this word
            if len(word_frames) > 1:
                # Limit to at most 9 frames for the grid
                display_frames = word_frames[:9]
                display_indices = frame_indices[:9]
                
                rows = min(3, (len(display_frames) + 2) // 3)
                cols = min(3, (len(display_frames) + rows - 1) // rows)
                
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
                fig.suptitle(f"Word: '{word}' - {len(display_frames)} frames", fontsize=16)
                
                # Handle case of single row or column
                if rows == 1 and cols == 1:
                    axes.imshow(display_frames[0], cmap='gray')
                    axes.set_title(f"Frame {display_indices[0]}")
                    axes.axis('off')
                elif rows == 1:
                    for i, (frame, frame_idx) in enumerate(zip(display_frames, display_indices)):
                        if i < cols:
                            axes[i].imshow(frame, cmap='gray')
                            axes[i].set_title(f"Frame {frame_idx}")
                            axes[i].axis('off')
                else:
                    for i, (frame, frame_idx) in enumerate(zip(display_frames, display_indices)):
                        if i < rows * cols:
                            row, col = i // cols, i % cols
                            axes[row, col].imshow(frame, cmap='gray')
                            axes[row, col].set_title(f"Frame {frame_idx}")
                            axes[row, col].axis('off')
                
                # Hide empty subplots
                for i in range(len(display_frames), rows * cols):
                    row, col = i // cols, i % cols
                    if rows > 1:
                        axes[row, col].axis('off')
                
                plt.tight_layout()
                clean_word = ''.join(c if c.isalnum() else '_' for c in word)
                grid_path = os.path.join(
                    OUTPUT_DIR, 
                    f"{speaker_id}_{video_name}_{clean_word}_grid.png"
                )
                plt.savefig(grid_path)
                plt.close()
                
                print(f"Saved {word} grid to {grid_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {speaker_id}/{video_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
# Add this function to visualize both the original and processed mouth regions
def visualize_processing(speaker_id, video_name):
    """Visualize the mouth processing to check for distortion"""
    print(f"Visualizing processing for {speaker_id}/{video_name}")
    
    video_path = os.path.join(VIDEO_DIR, speaker_id, f"{video_name}.mpg")
    frames = processor.extract_frames(video_path)
    
    # Process a few sample frames
    for i, frame in enumerate(frames[:5]):  # Process first 5 frames
        # Original frame
        original = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = processor.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print(f"No face detected in frame {i}")
            continue
            
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # More precise mouth region estimation
        mouth_y = y + int(h * 0.68)
        mouth_h = int(h * 0.32)
        mouth_x = x + int(w * 0.25)
        mouth_w = int(w * 0.50)
        
        # Extract mouth region
        mouth_roi = original[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
        
        # Process the frame
        processed = processor.process_frame(frame)
        
        # Create visualization
        plt.figure(figsize=(12, 4))
        
        # Original mouth region
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB))
        plt.title("Original Mouth Region")
        plt.axis('off')
        
        # Processed mouth region
        plt.subplot(1, 2, 2)
        plt.imshow(processed, cmap='gray')
        plt.title("Processed Mouth Region")
        plt.axis('off')
        
        # Save the comparison
        output_path = os.path.join(OUTPUT_DIR, f"{speaker_id}_{video_name}_frame{i}_comparison.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved comparison to {output_path}")

# Add a function to visualize model predictions
def visualize_predictions(speaker_id, video_name):
    """Visualize model predictions on a sample video"""
    print(f"Visualizing predictions for {speaker_id}/{video_name}")
    
    # Load the latest model checkpoint
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.h5')]
    if not checkpoints:
        print("No model checkpoints found")
        return False
    
    # Sort checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    print(f"Loading model from {latest_checkpoint}")
    
    # Load vocabulary
    texts = []
    text_path = os.path.join(TEXT_DIR, speaker_id, f"{video_name}.align")
    with open(text_path, 'r') as f:
        texts.append(f.read().strip())
    
    # Create vocabulary
    vocab_size = data_loader.create_vocab(texts)
    
    # Create model
    model = LipReader(vocab_size, MAX_SEQUENCE_LENGTH)
    
    # Compile model
    model.compile(optimizer='adam', metrics=['accuracy'])
    
    # Load weights
    model.load_weights(latest_checkpoint)
    
    # Load and process video
    frames, frame_labels = data_loader.load_sample(speaker_id, video_name)
    
    # Prepare input for model
    input_data = np.array(frames)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    # Get model predictions
    predictions = model.predict(input_data)
    
    # Convert predictions to words
    predicted_indices = np.argmax(predictions[0], axis=1)
    predicted_words = [data_loader.idx_to_char.get(idx, '?') for idx in predicted_indices]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(np.hstack(frames[:5]), cmap='gray')
    plt.title("Sample Frames")
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.text(0.1, 0.5, f"Actual: {' '.join(frame_labels[:20])}", fontsize=12)
    plt.text(0.1, 0.3, f"Predicted: {' '.join(predicted_words[:20])}", fontsize=12)
    plt.axis('off')
    
    # Save the visualization
    output_path = os.path.join(OUTPUT_DIR, f"{speaker_id}_{video_name}_prediction.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved prediction visualization to {output_path}")
    return True

# Choose a speaker and videos
speaker_id = TARGET_SPEAKER
try:
    # Get videos from the speaker
    video_files = [f[:-4] for f in os.listdir(os.path.join(VIDEO_DIR, speaker_id)) if f.endswith('.mpg')]
    if not video_files:
        print(f"No videos found for speaker {speaker_id}")
        exit(1)
    
    # Process just one video
    if video_files:
        # Choose the first video
        video_name = video_files[0]
        print(f"Processing single video: {speaker_id}/{video_name}")
        
        # Generate word examples
        success = generate_word_examples(speaker_id, video_name)
        
        # Also visualize the processing to check for distortion
        visualize_processing(speaker_id, video_name)
        
        # Try to visualize model predictions if checkpoints exist
        try:
            visualize_predictions(speaker_id, video_name)
        except Exception as e:
            print(f"Could not visualize predictions: {str(e)}")
        
        if success:
            print(f"Successfully processed video. Word examples saved to {OUTPUT_DIR}")
        else:
            print(f"Failed to process video {speaker_id}/{video_name}")
            
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

