import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os

class VideoProcessor:
    def __init__(self, target_size=(64, 64), fps=25):
        self.target_size = target_size
        self.fps = fps
        self.current_video = None
        
        # Load Haar cascade for face detection instead of MediaPipe
        # This avoids the timestamp issues completely
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def extract_frames(self, video_path):
        """Extract frames from video file with downsampling"""
        self.current_video = video_path
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Downsample frames to avoid processing too many
            sampling_rate = max(1, frame_count // 100)
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sampling_rate == 0:
                    frames.append(frame)
                frame_idx += 1
                    
            cap.release()
            return frames
        except Exception as e:
            raise RuntimeError(f"Error extracting frames from {video_path}: {str(e)}")
        finally:
            self.current_video = None
    
    def process_frame(self, frame):
        """Process a single frame to extract mouth region with focus on lips and chin"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
                
            # Get the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # More precise mouth region estimation (lower part of face, centered horizontally)
            mouth_y = y + int(h * 0.68)  # Start at 68% down the face
            mouth_h = int(h * 0.32)      # Take 32% of face height (taller to include more chin)
            
            # Narrow the width to focus more on the mouth
            mouth_x = x + int(w * 0.25)  # Start 25% from the left edge
            mouth_w = int(w * 0.50)      # Take 50% of face width
            
            # Ensure we don't go out of bounds
            mouth_y = max(0, mouth_y)
            mouth_h = min(frame.shape[0] - mouth_y, mouth_h)
            mouth_x = max(0, mouth_x)
            mouth_w = min(frame.shape[1] - mouth_x, mouth_w)
            
            # Extract mouth region with tighter crop
            mouth_roi = frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
            
            # Convert to grayscale
            mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            
            # Preserve aspect ratio while resizing
            h, w = mouth_gray.shape
            target_w, target_h = self.target_size
            
            # Calculate scaling factor to fit within target size while preserving aspect ratio
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize with proper aspect ratio
            resized = cv2.resize(mouth_gray, (new_w, new_h))
            
            # Create a black canvas of target size
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Calculate position to center the mouth in the canvas
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            # Place the resized mouth in the center of the canvas
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Enhance contrast to make lip features more visible
            canvas = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(canvas)
            
            # Normalize pixel values to [0, 1]
            processed = canvas.astype(np.float32) / 255.0
            
            return processed
            
        except Exception as e:
            if self.current_video:
                print(f"Error processing frame in video {self.current_video}")
            raise RuntimeError(f"Frame processing error: {str(e)}")

    def visualize_mouth_detection(self, video_path, output_dir="debug_images"):
        """Visualize mouth detection on sample frames and save images"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract video name for output files
        video_name = os.path.basename(video_path).split('.')[0]
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Process a few sample frames
        for i, frame in enumerate(frames[:5]):  # Process first 5 frames
            # Original frame
            original = frame.copy()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
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
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # More precise mouth region estimation
            mouth_y = y + int(h * 0.68)
            mouth_h = int(h * 0.32)
            mouth_x = x + int(w * 0.25)
            mouth_w = int(w * 0.50)
            
            # Draw mouth rectangle
            cv2.rectangle(frame, (mouth_x, mouth_y), (mouth_x+mouth_w, mouth_y+mouth_h), (0, 0, 255), 2)
            
            # Extract mouth region
            mouth_roi = original[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
            
            # Process the frame
            processed = self.process_frame(frame)
            
            # Create figure with 3 subplots
            plt.figure(figsize=(15, 5))
            
            # Original image with face and mouth rectangles
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Face and Mouth Detection")
            plt.axis('off')
            
            # Extracted mouth region
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB))
            plt.title("Extracted Mouth Region")
            plt.axis('off')
            
            # Processed mouth image
            plt.subplot(1, 3, 3)
            plt.imshow(processed, cmap='gray')
            plt.title("Processed Mouth (Model Input)")
            plt.axis('off')
            
            # Save figure
            output_path = os.path.join(output_dir, f"{video_name}_frame_{i}.png")
            plt.savefig(output_path)
            plt.close()
            
            print(f"Saved visualization to {output_path}") 