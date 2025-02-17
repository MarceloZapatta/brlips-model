import tensorflow as tf
import numpy as np
from utils.data_loader import DataLoader
from model.lip_reader import LipReader

def test_model(model_path, test_video_path, test_text_path):
    # Load the trained model
    model = LipReader(vocab_size)  # You'll need to use the same vocab_size as training
    model.load_weights(model_path)
    
    # Initialize data loader
    data_loader = DataLoader("", "")  # Directories not needed for single test
    
    # Load and process test video
    frames, reference_text = data_loader.load_sample(test_video_path)
    frames = np.expand_dims(frames, 0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(frames)
    predicted_sequence = np.argmax(predictions[0], axis=1)
    predicted_text = data_loader.sequence_to_text(predicted_sequence)
    
    # Calculate metrics
    cer = calculate_character_error_rate(reference_text, predicted_text)
    wer = calculate_word_error_rate(reference_text, predicted_text)
    
    print(f"Reference text: {reference_text}")
    print(f"Predicted text: {predicted_text}")
    print(f"Character Error Rate: {cer:.2f}")
    print(f"Word Error Rate: {wer:.2f}")

def calculate_character_error_rate(reference, hypothesis):
    # Levenshtein distance implementation
    return levenshtein_distance(reference, hypothesis) / len(reference)

def calculate_word_error_rate(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1] 