import cv2
import numpy as np
from mediapipe import solutions
import mediapipe as mp

class VideoProcessor:
    def __init__(self, target_size=(64, 64), fps=25):
        self.target_size = target_size
        self.fps = fps
        # Initialize face mesh detector
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Configure MediaPipe graph to use immediate input handler
        mp.solutions.mediapipe_options = {
            'input_stream_handler': 'ImmediateInputStreamHandler'
        }
        self.current_video = None  # Add this to track current video
        
    def extract_frames(self, video_path):
        """Extract frames from video file"""
        self.current_video = video_path  # Track current video being processed
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            return frames
        except Exception as e:
            raise RuntimeError(f"Error extracting frames from {video_path}: {str(e)}")
        finally:
            self.current_video = None
    
    def detect_mouth(self, frame):
        """Detect mouth region using MediaPipe Face Mesh"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        # Mouth landmarks indices
        MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375]
        landmarks = results.multi_face_landmarks[0]
        
        # Get mouth coordinates
        h, w = frame.shape[:2]
        mouth_points = []
        for idx in MOUTH_INDICES:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            mouth_points.append([x, y])
            
        mouth_points = np.array(mouth_points)
        x, y, w, h = cv2.boundingRect(mouth_points)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        
        return (x, y, w, h)
    
    def process_frame(self, frame):
        """Process single frame: detect mouth, crop, resize and apply filters"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return None
                
            # Mouth landmarks indices
            MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375]
            landmarks = results.multi_face_landmarks[0]
            
            # Get mouth coordinates
            h, w = frame.shape[:2]
            mouth_points = []
            for idx in MOUTH_INDICES:
                landmark = landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                mouth_points.append([x, y])
                
            mouth_points = np.array(mouth_points)
            x, y, w, h = cv2.boundingRect(mouth_points)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2*padding)
            h = min(frame.shape[0] - y, h + 2*padding)
            
            mouth_roi = frame[y:y+h, x:x+w]
            mouth_roi = cv2.resize(mouth_roi, self.target_size)
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            mouth_roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(mouth_roi)
            return mouth_roi / 255.0
            
        except Exception as e:
            if self.current_video:
                print(f"Error processing frame in video {self.current_video}")
            raise RuntimeError(f"MediaPipe frame processing error: {str(e)}") 