import cv2
import tensorflow as tf
import numpy as np
from typing import List
import time
import os
import matplotlib.pyplot as plt
try:
    # Try to import from the submodule path
    from infrastructure.model.src.vocab import char_to_num
except ImportError:
    # Fall back to importing from the base folder
    from vocab import char_to_num


def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str):
    path = path.numpy().decode("utf-8")
    file_name = path.split('/')[-1].split('.')[0]
    extension = path.split('/')[-1].split('.')[1]
    speaker_id = tf.convert_to_tensor(path).numpy().decode('utf-8').split('/')[-2]
    video_path = os.path.join('data',speaker_id,f'{file_name}.{extension}')
    alignment_path = os.path.join('data','alignments',speaker_id,f'{file_name}.align')
    print(f'this is the video path: {video_path}')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments

def load_video_for_prediction(path:str) -> List[float]:    
    cap = cv2.VideoCapture(path)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    face_detection_time = 0
    frame_extraction_time = 0

    for i in range(total_frames): 
        frame_start = time.time()
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {i}")
            continue

        frame_count += 1

        frame = tf.image.rgb_to_grayscale(frame)
        
        # Convert to grayscale for face detection
        # Convert TensorFlow tensor to NumPy array for OpenCV processing
        frame_np = frame.numpy()
        # Remove the extra dimension for grayscale since we already have grayscale
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
            print(f"No face detected in frame {i}")
            continue
        
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
            
        # Calculate mouth region coordinates
        mouth_y = y + int(h * 0.72)  # Start at 68% down the face
        mouth_h = int(h * 0.24)      # Take 32% of face height (taller to include more chin)
        mouth_x = x + int(w * 0.25)  # Start 25% from the left edge
        mouth_w = int(w * 0.50)      # Take 50% of face width
        
        # Calculate center of mouth region
        center_x = mouth_x + mouth_w // 2
        center_y = mouth_y + mouth_h // 2
        
        # Calculate ROI boundaries based on target size
        # roi_width = 240
        roi_width = 120
        # roi_height = 92
        roi_height = 46
        half_width, half_height = roi_width // 2, roi_height // 2
        
        # Calculate ROI coordinates
        x_start = max(0, center_x - half_width)
        y_start = max(0, center_y - half_height)
        x_end = min(frame.shape[1], center_x + half_width)
        y_end = min(frame.shape[0], center_y + half_height)
        
        mouth_roi = frame[y_start:y_end, x_start:x_end]
        
        mouth_roi = tf.image.resize(mouth_roi, (46, 120))
        
        # mouth_roi = frame[190:236,80:220,:] # 46x120
        frames.append(mouth_roi)
        frame_extraction_time += time.time() - frame_start
    
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    
    return tf.cast((frames - mean), tf.float32) / std

def load_video(path:str) -> List[float]: 
    start_time = time.time()
    
    cap = cv2.VideoCapture(path)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    frames = []
    
    
    parts = path.split('/')
    for part in parts:
        if part.startswith('s') and len(part) > 1 and part[1:].isdigit():
            subject_id = part
            break
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Starting to process {path} with {total_frames} frames")

    frame_count = 0
    face_detection_time = 0
    frame_extraction_time = 0

    for i in range(total_frames): 
        frame_start = time.time()
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {i}")
            continue

        frame_count += 1

        # Fix video orientation
        file_name = path.split('/')[-1].split('.')[0]
        
        # if (int(file_name) <= 100):
        #     frame = cv2.flip(frame, 0)

        frame = tf.image.rgb_to_grayscale(frame)
        
        # Convert to grayscale for face detection
        # Convert TensorFlow tensor to NumPy array for OpenCV processing
        frame_np = frame.numpy()
        # Remove the extra dimension for grayscale since we already have grayscale
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
            print(f"No face detected in frame {i}")
            continue
        
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
            
        # Calculate mouth region coordinates
        mouth_y = y + int(h * 0.72)  # Start at 68% down the face
        mouth_h = int(h * 0.24)      # Take 32% of face height (taller to include more chin)
        mouth_x = x + int(w * 0.25)  # Start 25% from the left edge
        mouth_w = int(w * 0.50)      # Take 50% of face width
        
        # Calculate center of mouth region
        center_x = mouth_x + mouth_w // 2
        center_y = mouth_y + mouth_h // 2
        
        # Calculate ROI boundaries based on target size
        # roi_width = 240
        roi_width = 120
        # roi_height = 92
        roi_height = 46
        half_width, half_height = roi_width // 2, roi_height // 2
        
        # Calculate ROI coordinates
        x_start = max(0, center_x - half_width)
        y_start = max(0, center_y - half_height)
        x_end = min(frame.shape[1], center_x + half_width)
        y_end = min(frame.shape[0], center_y + half_height)
        
        mouth_roi = frame[y_start:y_end, x_start:x_end]
        
        mouth_roi = tf.image.resize(mouth_roi, (46, 120))
        
        # mouth_roi = frame[190:236,80:220,:] # 46x120
        frames.append(mouth_roi)
        frame_extraction_time += time.time() - frame_start
        
        os.makedirs(f'preprocessed_data_en/{subject_id}/{file_name}', exist_ok=True)
        example_filename = f'preprocessed_data_en/{subject_id}/{file_name}/{i}_example.png'
        plt.imsave(example_filename, mouth_roi)
    
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    
    total_time = time.time() - start_time
    print(f"Processed {path}: {frame_count}/{total_frames} frames in {total_time:.2f}s")
    print(f"Face detection: {face_detection_time:.2f}s, Frame processing: {frame_extraction_time:.2f}s")
    print(f"Average per frame: {frame_extraction_time/max(1, frame_count):.4f}s")
    
    return tf.cast((frames - mean), tf.float32) / std