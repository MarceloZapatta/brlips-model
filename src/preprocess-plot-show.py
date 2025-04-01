import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from tqdm import tqdm

def visualize_preprocessing(video_path, output_dir):
    """
    Visualizes each step of the preprocessing pipeline for a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save visualization images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Load face cascade classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Get video details
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing {video_path} with {total_frames} frames at {fps} FPS")
    
    # Initialize counters
    frame_count = 0
    face_detection_time = 0
    
    # Process frames
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {frame_idx}")
            continue

        frame_count += 1
        
        # Store original frame for visualization
        original_frame = frame.copy()
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert BGR to RGB for matplotlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to TensorFlow tensor and grayscale
        tf_frame = tf.convert_to_tensor(frame)
        tf_gray_frame = tf.image.rgb_to_grayscale(tf_frame)
        
        # Convert TensorFlow tensor to NumPy array for OpenCV processing
        frame_np = tf_gray_frame.numpy()
        # Remove the extra dimension for grayscale
        frame_np = np.squeeze(frame_np)
        
        face_start = time.time()
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            frame_np, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        face_detection_time += time.time() - face_start
        
        if len(faces) == 0:
            # Skips to next frame if no face is detected
            print(f"No face detected in frame {frame_idx}")
            continue
        
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Create a copy of the RGB frame with face rectangle drawn
        combined_visualization = gray_frame.copy()
        cv2.rectangle(combined_visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Calculate mouth region coordinates
        mouth_y = y + int(h * 0.68)  # Start at 72% down the face
        mouth_h = int(h * 0.32)      # Take 24% of face height
        mouth_x = x + int(w * 0.25)  # Start 25% from the left edge
        mouth_w = int(w * 0.50)      # Take 50% of face width
        
        # Draw mouth region rectangle on the combined visualization
        cv2.rectangle(combined_visualization, (mouth_x, mouth_y), 
                     (mouth_x+mouth_w, mouth_y+mouth_h), (255, 0, 0), 2)
        
        # Calculate center of mouth region
        center_x = mouth_x + mouth_w // 2
        center_y = mouth_y + mouth_h // 2
        
        # Calculate ROI boundaries based on target size
        roi_width = 240
        # roi_width = 120
        roi_height = 92
        half_width, half_height = roi_width // 2, roi_height // 2
        
        # Calculate ROI coordinates
        x_start = max(0, center_x - half_width)
        y_start = max(0, center_y - half_height)
        x_end = min(frame.shape[1], center_x + half_width)
        y_end = min(frame.shape[0], center_y + half_height)
        
        # Draw final ROI rectangle on the combined visualization
        cv2.rectangle(combined_visualization, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        
        # Extract the mouth ROI
        mouth_roi = tf_gray_frame[y_start:y_end, x_start:x_end]
        
        # Resize the ROI to target dimensions
        resized_roi = tf.image.resize(mouth_roi, (46, 120))
        
        # Convert tensors to numpy for visualization
        mouth_roi_np = mouth_roi.numpy()
        resized_roi_np = resized_roi.numpy()
        
        # Perform normalization on this single frame (for demonstration)
        # Calculate mean and std for this ROI
        roi_mean = tf.math.reduce_mean(resized_roi)
        roi_std = tf.math.reduce_std(tf.cast(resized_roi, tf.float32))
        
        # Normalize the ROI
        normalized_roi = tf.cast((resized_roi - roi_mean), tf.float32) / roi_std
        normalized_roi_np = normalized_roi.numpy()
        
        # Create figure for visualization - extremely compact for paper
        fig = plt.figure(figsize=(10, 2.2))  # Keep the same height
        
        # Create grid for images only - absolute minimum spacing with maximum top position
        gs = GridSpec(1, 5, figure=fig, wspace=0.01, top=0.95, bottom=0.01, left=0.01, right=0.99)
        
        # Add titles in a single line
        titles = ['1. Original', '2. "Grayscale"', '3. Detecção da face', 
                 '4. Recorte dos lábios', '5. Normalização']
        
        # Create subplots and add images WITHOUT titles
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb_frame)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gray_frame, cmap='gray')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(combined_visualization, cmap='gray')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(np.squeeze(mouth_roi_np), cmap='gray')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(np.squeeze(normalized_roi_np), cmap='gray')
        ax5.axis('off')
        
        # Add titles with fixed positions
        # Calculate positions with fixed gaps
        title_positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # Evenly spaced positions
        
        # Create a separate subplot for titles - positioned almost on top of the images
        title_ax = plt.axes([0, 0.8, 1, 0.1], frameon=False)  # Extremely close to images
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        
        # Add titles at fixed positions
        for pos, title in zip(title_positions, titles):
            title_ax.text(pos, 0.5, title, ha='center', va='center', fontsize=7)
        
        # Save the figure with zero padding
        output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}_paper.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.0)  # No padding
        plt.close(fig)
        
        # Only process a few frames to avoid generating too many images
        if frame_idx >= 1:  # Process only first 2 frames
            break
    
    cap.release()
    print(f"Visualization complete. Images saved to {output_dir}")
    print(f"Processed {frame_count} frames. Average face detection time: {face_detection_time/frame_count:.4f}s per frame")

# Example usage
video_path = "data/s2/1.mp4"  # Change this to your video path
output_dir = "visualization_results"

visualize_preprocessing(video_path, output_dir)
