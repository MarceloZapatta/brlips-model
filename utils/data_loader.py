import os
import numpy as np
from .video_processor import VideoProcessor

class DataLoader:
    def __init__(self, video_dir, text_dir, max_sequence_length=100):
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.max_sequence_length = max_sequence_length
        self.video_processor = VideoProcessor()
        self.char_to_idx = None
        self.idx_to_char = None
        
    def create_vocab(self, texts):
        """Create character vocabulary from all text samples"""
        chars = set(''.join(texts))
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        return len(self.char_to_idx)
        
    def text_to_sequence(self, text):
        """Convert text to integer sequence"""
        return [self.char_to_idx[char] for char in text]
        
    def sequence_to_text(self, sequence):
        """Convert integer sequence back to text"""
        return ''.join([self.idx_to_char[idx] for idx in sequence])
        
    def load_sample(self, video_name):
        """Load and process single video-text pair"""
        # Load video
        video_path = os.path.join(self.video_dir, video_name + '.mp4')
        frames = self.video_processor.extract_frames(video_path)
        
        # Process frames
        processed_frames = []
        for frame in frames:
            processed = self.video_processor.process_frame(frame)
            if processed is not None:
                processed_frames.append(processed)
                
        # Pad or truncate sequence
        if len(processed_frames) > self.max_sequence_length:
            processed_frames = processed_frames[:self.max_sequence_length]
        else:
            pad_length = self.max_sequence_length - len(processed_frames)
            processed_frames.extend([np.zeros_like(processed_frames[0])] * pad_length)
            
        # Load and process text
        text_path = os.path.join(self.text_dir, video_name + '.txt')
        with open(text_path, 'r') as f:
            text = f.read().strip()
            
        return np.array(processed_frames), text 