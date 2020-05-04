import tensorflow as tf
import numpy as np
from rnn import build_model
import os


def split_input_target(chunk):
    ''' Creates an input and target example for each sequence'''
    input_text = chunk[:-1]  # Removes the last character
    target_text = chunk[1:]  # Removes the first character
    return input_text, target_text


def loss(labels, logits):
    ''' Performs Sparce Caterorical Crossentropy Loss '''
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# Read text
path_to_file = '../data/hp1.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# Build a vocabulary of unique characters in the text
vocab = sorted(set(text))

# Map each unique char to a different index
char2idx = {u: i for i, u in enumerate(vocab)}
# Map the index to the respective char
idx2char = np.array(vocab)
# Convert all the text to indices
text_as_int = np.array([char2idx[c] for c in text])

# Maximum length sentence we want for a single input
seq_length = 100

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# Get inputs and targets for each sequence
dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000
# Suffle the dataset and get batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build the model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
model = build_model(vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    rnn_units=rnn_units,
                    batch_size=BATCH_SIZE)
model.compile(optimizer='adam', loss=loss)

# Define checkpoint path for each batch
checkpoint_path = "../training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True)

# Number of epochs (full training pass over the entire dataset)
EPOCHS = 1
# Train the model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
