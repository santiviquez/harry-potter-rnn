import tensorflow as tf

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    ''' Builds a simple sequencial 3 layers model '''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)])
    return model