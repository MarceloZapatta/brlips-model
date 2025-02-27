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
        self.fps = 25  # Standard video FPS
        self.problematic_videos = set()  # Track videos that cause errors
        
    def create_vocab(self, texts):
        """Create character vocabulary from all text samples"""
        # Split texts into words and get unique words including 'sil'
        words = set()
        for text in texts:
            for line in text.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    word = parts[2]
                    words.add(word)
        words.add('sil')  # Add silence token
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(words))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Print vocabulary for debugging
        print(f"Created vocabulary with {len(words)} words:")
        print(f"First 10 words: {list(self.word_to_idx.keys())[:10]}")
        print(f"Word to index mapping (sample): {dict(list(self.word_to_idx.items())[:5])}")
        print(f"Index to word mapping (sample): {dict(list(self.idx_to_word.items())[:5])}")
        
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
        
    def parse_align_file(self, text, total_frames):
        """Parse align file content into timing and text
        
        Args:
            text: Content of the alignment file
            total_frames: Total number of frames in the video for scaling
        
        Returns:
            List of alignment dictionaries with start_frame, end_frame, and text
        """
        alignments = []
        lines = text.strip().split('\n')
        
        # Get the total duration from the last line's end time
        if lines:
            last_line = lines[-1].strip().split()
            if len(last_line) >= 2:
                total_duration_ms = int(last_line[1])
            else:
                total_duration_ms = 0
        else:
            total_duration_ms = 0
        
        # If we have a valid duration, use it to scale frame indices
        if total_duration_ms > 0 and total_frames > 0:
            ms_per_frame = total_duration_ms / total_frames
        else:
            # Fallback to standard 25fps (40ms per frame)
            ms_per_frame = 1000 / self.fps
        
        for line in text.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) == 3:
                start_ms, end_ms, word = parts
                
                # Convert milliseconds to frame numbers using the calculated ms_per_frame
                start_frame = int(int(start_ms) / ms_per_frame)
                end_frame = int(int(end_ms) / ms_per_frame)
                
                # Ensure end_frame doesn't exceed total_frames
                if end_frame > total_frames:
                    end_frame = total_frames
                    
                # Only add if end_frame > start_frame to avoid empty segments
                if end_frame > start_frame:
                    alignments.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'text': word
                    })
        
        return alignments

    def load_sample(self, speaker_id, video_name, augment=False):
        """Load and process single video-text pair"""
        # Validate file existence
        video_path = os.path.join(self.video_dir, speaker_id, f"{video_name}.mpg")
        text_path = os.path.join(self.text_dir, speaker_id, f"{video_name}.align")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found: {text_path}")
        
        # Skip known problematic videos
        video_key = f"{speaker_id}/{video_name}"
        if video_key in self.problematic_videos:
            raise ValueError(f"Skipping known problematic video: {video_key}")
        
        try:
            # Load video
            frames = self.video_processor.extract_frames(video_path)
            
            # Process frames with error handling for individual frames
            processed_frames = []
            for i, frame in enumerate(frames):
                try:
                    processed = self.video_processor.process_frame(frame)
                    if processed is not None:
                        processed_frames.append(processed)
                except Exception as e:
                    print(f"Warning: Error processing frame {i} in {video_path}: {str(e)}")
                    # Continue with other frames instead of failing completely
                    continue
            
            if not processed_frames:
                self.problematic_videos.add(video_key)
                raise ValueError(f"No valid frames processed from video: {video_key}")
            
            # Load and parse alignment file
            with open(text_path, 'r') as f:
                align_content = f.read()
                # Pass the total number of processed frames to properly scale alignments
                alignments = self.parse_align_file(align_content, len(processed_frames))
            
            # Create frame-level text labels
            frame_labels = []
            
            for i in range(len(processed_frames)):
                # Find the alignment that contains this frame
                word_found = False
                for alignment in alignments:
                    if i >= alignment['start_frame'] and i < alignment['end_frame']:
                        frame_labels.append(alignment['text'])
                        word_found = True
                        break
                
                # If no alignment contains this frame, it's silence
                if not word_found:
                    frame_labels.append('sil')
            
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
            # Mark this video as problematic for future reference
            self.problematic_videos.add(video_key)
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error processing {video_key}: {str(e)}") 

    @property
    def idx_to_char(self):
        """Alias for idx_to_word for compatibility with LipReader"""
        return self.idx_to_word if hasattr(self, 'idx_to_word') else None