import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Reshape, Bidirectional, LSTM, Dropout, Dense
from vocab import char_to_num

def load_model_en_with_weights():
    model = create_model_en()
    return model.load_weights('models/complete_models/en/1/epoch_100.weights.h5')

def load_model_br_with_weights():
    model = create_model_br()
    return model.load_weights('models/complete_models/pt_br/720p/epoch_100.weights.h5')

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