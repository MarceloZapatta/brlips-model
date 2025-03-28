import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Reshape, Bidirectional, LSTM, Dropout, Dense
try:
    # Try to import from the submodule path
    from infrastructure.model.src.vocab import char_to_num, num_to_char
except ImportError:
    # Fall back to importing from the base folder
    from vocab import char_to_num, num_to_char
import os

def load_model_en_with_weights():
    model = create_model_en()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'models', 'checkpoint', 'en', '1', 'epoch_100.weights.h5')
    model.load_weights(weights_path)
    return model

def load_model_br_with_weights():
    model = create_model_br()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'models', 'checkpoint', 'pt_br', '720p', 'epoch_100.weights.h5')
    model.load_weights(weights_path)
    return model

def create_model_en():
    max_frames = 75
    height = 46
    width = 120
    model = Sequential([
      Input(shape=(max_frames, height, width, 1)),
      Conv3D(128, 3, padding='same', activation='relu'),
      MaxPool3D((1,2,2)),
      Conv3D(256, 3, padding='same', activation='relu'),
      MaxPool3D((1,2,2)),
      Conv3D(max_frames, 3, padding='same', activation='relu'),
      MaxPool3D((1,2,2)),
      Reshape((max_frames, (height // 8) * (width // 8) * max_frames)),
      Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
      Dropout(0.5),
      Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
      Dropout(0.5),
      Dense(char_to_num.vocabulary_size() + 1, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer='zeros',
        activation='softmax')
      ])
    return model

def create_model_br():
    max_frames = 160
    height = 46
    width = 120
    model = Sequential([
      Input(shape=(max_frames, height, width, 1)),
      Conv3D(128, 3, padding='same', activation='relu'),
      MaxPool3D((1,2,2)),
      Conv3D(256, 3, padding='same', activation='relu'),
      MaxPool3D((1,2,2)),
      Conv3D(max_frames, 3, padding='same', activation='relu'),
      MaxPool3D((1,2,2)),
      Reshape((max_frames, (height // 8) * (width // 8) * max_frames)),
      Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
      Dropout(0.5),
      Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
      Dropout(0.5),
      Dense(char_to_num.vocabulary_size() + 1, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer='zeros',
        activation='softmax')
      ])
    return model

def predict(frames) -> str:
    model = load_model_en_with_weights()
    print(f"Model loaded from {model}")
    frames = tf.expand_dims(frames, axis=0)
    print(f"Frames: {frames.shape}")
    predictions = model.predict(frames)

    seq_length = tf.shape(predictions)[1]

    input_length = tf.ones((1,), dtype=tf.int32) * seq_length

    decoded = tf.keras.backend.ctc_decode(predictions, input_length=input_length, greedy=True)[0][0].numpy()

    prediction_text = tf.strings.reduce_join([num_to_char(word) for word in decoded]).numpy().decode('utf-8')
    prediction_text = ' '.join(prediction_text.split())

    return prediction_text
