import tensorflow as tf
import numpy as np
import os


def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    '''Generates text using the learned model'''

    # Converting our start string to numbers
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our result
    text_generated = []
    # Resets the state of metrics
    model.reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


path_to_file = 'data/hp1.txt'
# Read text
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# Unique characters in the text
vocab = sorted(set(text))

# Map each unique char to a different index
char2idx = {u: i for i, u in enumerate(vocab)}
# Map the index to the respective char
idx2char = np.array(vocab)