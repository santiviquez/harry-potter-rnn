import numpy as np
import tensorflow as tf

# First we are going to load the data
with open('../data/input.txt', 'r') as f:
    text = f.read()

# Couple of dictionaries to convert the characters to and from integers
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

# Encode each character as integer so it will be easier to feed our network
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

def get_batches(arr, batch_size, n_steps):
    # Total number per batch
    characters_per_batch = batch_size*n_steps

    # Total number of batches we can make
    n_batches = len(arr)//characters_per_batch

    # We want to keep just enough characters to make full batches
    arr = arr[:characters_per_batch*n_batches]

    # Lets reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # Window size is n_steps x batch_size
    for n in range(0, arr.shape[1], n_steps):
        # Features
        x = arr[:, n:n+n_steps]
        
        # Targets
        y_temp = arr[:, n+1:n+n_steps+1]
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:, :y_temp.shape[1]] = y_temp

        yield x, y

batches = get_batches(encoded, 3, 1)
x, y = next(batches)
