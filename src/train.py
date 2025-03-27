import tensorflow as tf
import numpy as np
from typing import List
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from jiwer import wer
from brlips_model import create_model_en

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    try:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4000)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
except:
    pass

def load_npz(file_path):
    file_path = file_path.numpy().decode("utf-8")
    print('training data:', file_path)
    data = np.load(file_path)
    frames = data['frames']
    alignments = data['alignments']
    
    # Cast to float32
    frames = tf.cast(frames, tf.float32)
    
    # Find min and max per batch to preserve contrast
    min_val = tf.reduce_min(frames)
    max_val = tf.reduce_max(frames)
    
    # Avoid division by zero
    denominator = tf.maximum(max_val - min_val, 1e-6)
    
    # Apply min-max scaling to get to [0,1] range
    frames = (frames - min_val) / denominator

    return frames, alignments

def mappable_function(path) -> List[str]:
    return tf.py_function(load_npz, [path], [tf.float32, tf.int64])

max_frames = 75

padded_shapes = ([max_frames, None, None, None], [40])
batch_size = 1
total_per_speaker = 200
test_size = 20
train_size = 180

data = tf.data.Dataset.list_files('./preprocessed_data_en/s1/*.npz')
data = data.shuffle(total_per_speaker, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(batch_size, padded_shapes=padded_shapes)
data = data.prefetch(tf.data.AUTOTUNE)

# Added for split 
train = data.take(train_size)
test = data.skip(train_size).take(test_size)

data2 = tf.data.Dataset.list_files('./preprocessed_data_en/s6/*.npz')
data2 = data2.shuffle(total_per_speaker, reshuffle_each_iteration=False)
data2 = data2.map(mappable_function)
data2 = data2.padded_batch(batch_size, padded_shapes=padded_shapes)
data2 = data2.prefetch(tf.data.AUTOTUNE)

# Added for split 
train2 = data2.take(train_size)
test2 = data2.skip(train_size).take(test_size)

train = train.concatenate(train2)
test = test.concatenate(test2)

initial_epoch = 0
height = 46
width = 120
channels = 1

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def scheduler(epoch, lr):
    # Initial high rate
    if epoch < 30:
        return 0.0001
    # Cyclical schedule to help escape plateaus
    cycle = (epoch // 5) % 3  # Cycles of 5 epochs, 3 different rates
    if cycle == 0:
        return 0.00001  # Lower rate
    elif cycle == 1:
        return 0.00005  # Medium rate
    else:
        return 0.000025  # Intermediate rate
    
schedule_callback = LearningRateScheduler(scheduler)
    
def CTCLoss(y_true, y_pred):
    print(y_true.shape, y_true)
    print(y_pred.shape, y_pred)
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
    print(batch_len)

    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    # Create tensors with the actual batch size
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    
    
    # Debug values
    tf.print("y_true sample:", y_true[0, :5])
    tf.print("y_pred sample (logits):", y_pred[0, 0, :5])


    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    tf.print("Input length:", input_length[0])
    
    tf.print("Label length:", label_length[0])
    
    # Debug the loss itself
    tf.print("CTC loss computed:", loss)
    
    return loss

class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super(SaveModelCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        model_save_path = os.path.join(self.save_path, f'epoch_{epoch + 1}.keras')
        self.model.save(model_save_path)
        print(f'\nModel saved to {model_save_path}')

save_model_callback = SaveModelCallback(save_path='models/complete_models/en/1')

model = create_model_en()

model.compile(optimizer=Adam(learning_rate=0.001), loss=CTCLoss)

checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint', 'en', '1', 'epoch_{epoch:02d}.weights.h5'), monitor='loss', save_weights_only=True) 

model.fit(train, validation_data=test, epochs=100, initial_epoch=initial_epoch, callbacks=[checkpoint_callback, schedule_callback, save_model_callback])

# Function to decode CTC outputs
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy decoding
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Convert to text
    output_texts = []
    for result in results:
        result = tf.strings.reduce_join([num_to_char(word) for word in result]).numpy().decode('utf-8')
        # Clean up the output text - removing consecutive spaces and trailing spaces
        result = ' '.join(result.split())
        output_texts.append(result)
    return output_texts

# Function to convert ground truth to text
def decode_ground_truth(labels):
    output_texts = []
    for label in labels:
        label = tf.strings.reduce_join([num_to_char(word) for word in label]).numpy().decode('utf-8')
        # Clean up - removing consecutive spaces and trailing spaces
        label = ' '.join(label.split())
        output_texts.append(label)
    return output_texts

# Calculate WER for the test dataset
num_test_batches = 10  # Limit to 10 batches to avoid long computation
all_predictions = []
all_ground_truth = []

print("Evaluating model performance...")
test_iter = test.as_numpy_iterator()

for i in range(min(num_test_batches, len(test))):
    try:
        test_sample = next(test_iter)
        X, y = test_sample
        
        # Make prediction
        batch_predictions = model.predict(X)
        
        # Decode predictions
        decoded_predictions = decode_batch_predictions(batch_predictions)
        all_predictions.extend(decoded_predictions)
        
        # Decode ground truth
        decoded_ground_truth = decode_ground_truth(y)
        all_ground_truth.extend(decoded_ground_truth)
        
        # Print sample comparisons
        if i < 3:  # Show first 3 samples
            print(f"\nSample {i+1}:")
            print(f"Ground truth: '{decoded_ground_truth[0]}'")
            print(f"Prediction  : '{decoded_predictions[0]}'")
    except StopIteration:
        break
    except Exception as e:
        print(f"Error processing batch {i}: {e}")
        continue

# Calculate overall WER
if all_predictions and all_ground_truth:
    error_rate = wer(all_ground_truth, all_predictions)
    print(f"\nWord Error Rate (WER): {error_rate:.4f}")
    
    # Calculate character error rate
    def cer(ground_truth, predictions):
        total_chars = sum(len(ref) for ref in ground_truth)
        total_edits = 0
        for pred, ref in zip(predictions, ground_truth):
            # Convert strings to tensor format for tf.edit_distance
            pred_tokens = tf.strings.unicode_split(pred, 'UTF-8')
            ref_tokens = tf.strings.unicode_split(ref, 'UTF-8')
            
            # Convert to sparse tensors
            pred_sparse = tf.sparse.from_dense(tf.expand_dims(pred_tokens, 0))
            ref_sparse = tf.sparse.from_dense(tf.expand_dims(ref_tokens, 0))
            
            # Calculate edit distance
            distance = tf.edit_distance(pred_sparse, ref_sparse, normalize=False)
            total_edits += distance.numpy()[0]
        return total_edits / max(total_chars, 1)
    
    char_error_rate = cer(all_ground_truth, all_predictions)
    print(f"Character Error Rate (CER): {char_error_rate:.4f}")
    
    # Count number of correct predictions (exact matches)
    correct = sum(pred == ref for pred, ref in zip(all_predictions, all_ground_truth))
    accuracy = correct / len(all_ground_truth) if all_ground_truth else 0
    print(f"Exact Match Accuracy: {accuracy:.4f} ({correct}/{len(all_ground_truth)})")
else:
    print("No predictions or ground truth available for evaluation")
    
# Save results to a text file
with open('speech_recognition_results.txt', 'w') as f:
    f.write("SPEECH RECOGNITION EVALUATION RESULTS\n")
    f.write("===================================\n\n")
    f.write(f"Word Error Rate (WER): {error_rate:.4f}\n")
    f.write(f"Character Error Rate (CER): {char_error_rate:.4f}\n")
    f.write(f"Exact Match Accuracy: {accuracy:.4f} ({correct}/{len(all_ground_truth)})\n\n")
    
    f.write("SAMPLE PREDICTIONS\n")
    f.write("=================\n\n")
    
    # Write some sample predictions (up to 10)
    num_samples = min(10, len(all_ground_truth))
    for i in range(num_samples):
        f.write(f"Sample {i+1}:\n")
        f.write(f"Ground truth: '{all_ground_truth[i]}'\n")
        f.write(f"Prediction  : '{all_predictions[i]}'\n\n")
    
    print(f"Results saved to speech_recognition_results.txt")