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
        self.fps = 25  # Standard video FPS
        
    def create_vocab(self, texts):
        """Create character vocabulary from all text samples"""
        # Split texts into words and get unique words including 'sil'
        words = set()
        for text in texts:
            for line in text.strip().split('\n'):
                _, _, word = line.strip().split()
                words.add(word)
        words.add('sil')  # Add silence token
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(words))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        return len(self.word_to_idx)
        
    def text_to_sequence(self, text, pad_to_length=None):
        """Convert text to a numerical sequence with optional padding
        
        Args:
            text: Input text or list of words
            pad_to_length: Optional length to pad sequence to
        
        Returns:
            List of indices with optional padding
        """
        # Handle single word or list of words
        if isinstance(text, str):
            sequence = [self.word_to_idx.get(text, 0)]
        else:
            sequence = [self.word_to_idx.get(word, 0) for word in text]
        
        # Apply padding if specified
        if pad_to_length is not None:
            if len(sequence) > pad_to_length:
                sequence = sequence[:pad_to_length]
            else:
                # Pad with index for 'sil' token
                sequence.extend([self.word_to_idx['sil']] * (pad_to_length - len(sequence)))
            
        return sequence
        
    def sequence_to_text(self, sequence):
        """Convert integer sequence back to text"""
        return ' '.join([self.idx_to_word[idx] for idx in sequence])
        
    def parse_align_file(self, text):
        """Parse align file content into timing and text"""
        alignments = []
        for line in text.strip().split('\n'):
            start, end, word = line.strip().split()
            alignments.append({
                'start_frame': int(int(start) * self.fps / 1000),  # Convert ms to frame number
                'end_frame': int(int(end) * self.fps / 1000),
                'text': word
            })
        return alignments

    def load_sample(self, speaker_id, video_name):
        """Load and process single video-text pair"""
        # Validate file existence
        video_path = os.path.join(self.video_dir, speaker_id, f"{video_name}.mpg")
        text_path = os.path.join(self.text_dir, speaker_id, f"{video_name}.align")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found: {text_path}")
        
        try:
            # Load video
            frames = self.video_processor.extract_frames(video_path)
            
            # Process frames
            processed_frames = []
            for i, frame in enumerate(frames):
                try:
                    processed = self.video_processor.process_frame(frame)
                    if processed is not None:
                        processed_frames.append(processed)
                except Exception as e:
                    raise RuntimeError(f"Error processing frame {i} in {video_path}: {str(e)}")
            
            if not processed_frames:
                raise ValueError(f"No valid frames processed from video: {speaker_id}/{video_name}")
            
            # Load and parse alignment file
            with open(text_path, 'r') as f:
                alignments = self.parse_align_file(f.read())
            
            # Create frame-level text labels
            frame_labels = []
            current_alignment_idx = 0
            
            for i in range(len(processed_frames)):
                while (current_alignment_idx < len(alignments) - 1 and 
                       i >= alignments[current_alignment_idx]['end_frame']):
                    current_alignment_idx += 1
                
                if (i >= alignments[current_alignment_idx]['start_frame'] and 
                    i < alignments[current_alignment_idx]['end_frame']):
                    frame_labels.append(alignments[current_alignment_idx]['text'])
                else:
                    frame_labels.append('sil')  # silence for frames between words
            
            # Pad or truncate sequence
            if len(processed_frames) > self.max_sequence_length:
                processed_frames = processed_frames[:self.max_sequence_length]
                frame_labels = frame_labels[:self.max_sequence_length]
            else:
                pad_length = self.max_sequence_length - len(processed_frames)
                processed_frames.extend([np.zeros_like(processed_frames[0])] * pad_length)
                frame_labels.extend(['sil'] * pad_length)
            
            return np.array(processed_frames), frame_labels
        
        except Exception as e:
            raise RuntimeError(f"Error processing {speaker_id}/{video_name}: {str(e)}") 