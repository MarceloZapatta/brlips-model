import os
import numpy as np
from tqdm import tqdm
import cv2
import tensorflow as tf
from typing import List
from matplotlib import pyplot as plt
import time
from video_loader import load_data

max_frame = 0;

data_path = 'preprocessed_data_en/s3'

# Create directories to store preprocessed data
os.makedirs(data_path, exist_ok=True)
# os.makedirs('preprocessed_data_original_3/s2', exist_ok=True)

folder_name = 'data/s3'
# List all files in the data directories
s1_paths = [f'./{folder_name}/{file}' for file in os.listdir(f'./{folder_name}/')]

# Process speaker 1
print("Processing speaker 1 videos...")

for path in tqdm(s1_paths):
    # Extract file name without extension
    file_name = os.path.basename(path).split('.')[0]
    output_path = f'{data_path}/{file_name}.npz'
    
    # Skip if already processed
    if os.path.exists(output_path):
        continue
    
    # Process video and get frames and alignments
    frames, alignments = load_data(tf.convert_to_tensor(path))
    
    if len(frames) > max_frame:
        max_frame = len(frames)
    
    # Save both frames and alignments in the same file
    np.savez(output_path, frames=frames.numpy(), alignments=alignments.numpy())


data_path = 'preprocessed_data_en/s6'

folder_name = 'data/s6'
s2_paths = [f'./{folder_name}/{file}' for file in os.listdir(f'./{folder_name}/')]

# Process speaker 2
print("Processing speaker 2 videos...")

for path in tqdm(s2_paths):
    # Extract file name without extension
    file_name = os.path.basename(path).split('.')[0]
    output_path = f'{data_path}/{file_name}.npz'
    
    # Skip if already processed
    if os.path.exists(output_path):
        continue
    
    # Process video and get frames and alignments
    frames, alignments = load_data(tf.convert_to_tensor(path))
    
    if len(frames) > max_frame:
        max_frame = len(frames)
    
    # Save both frames and alignments in the same file
    np.savez(output_path, frames=frames.numpy(), alignments=alignments.numpy())

print(f"Preprocessing completed. Data saved to '{output_path}' directory.")
print(f"Max frame count: {max_frame}")
