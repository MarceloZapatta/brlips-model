import tensorflow as tf
from tensorflow.keras import layers, Model
import keras

class LipReader(Model):
    def __init__(self, vocab_size, max_sequence_length=100):
        super(LipReader, self).__init__()
        
        # CNN for spatial features
        self.conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D()
        self.conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D()
        
        # Temporal features
        self.lstm1 = keras.layers.LSTM(
            256,
            return_sequences=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal'
        )
        self.lstm2 = keras.layers.LSTM(
            256,
            return_sequences=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal'
        )
        
        # Output layer
        self.dense = layers.Dense(vocab_size, activation='softmax')
        
    def call(self, inputs):
        # Input shape: (batch_size, sequence_length, height, width, channels)
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        
        # Reshape for CNN
        x = tf.reshape(inputs, (-1, 64, 64, 1))
        
        # CNN layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        
        # Reshape for LSTM
        # Calculate feature dimensions
        feature_dim = 8 * 8 * 128  # After 3 pooling layers: 64->32->16->8
        x = tf.reshape(x, (batch_size, sequence_length, feature_dim))
        
        # Ensure input is float32
        x = tf.cast(x, tf.float32)
        
        # LSTM layers
        x = self.lstm1(x)
        x = self.lstm2(x)
        
        # Output layer
        x = self.dense(x)
        
        return x 