import streamlit as st
import tensorflow as tf
import numpy as np
from model.predict_harry_potter import generate_text
from model.rnn import build_model
import os

# Streamlit app title
st.markdown('## **Harry Potter** *And The Deep Learning Experiment*')
st.markdown('')
st.markdown('Tired that there\'re only 7 Happy Potter books and 8 films? **Me too!** That\'s way I told my computer to create this new Harry Potter adventure.')

st.markdown('Go on and create yours! (Sometimes the story doesn\'t make a lot of sense but, who does?)')


# Number of characters to generata
num_generate = st.number_input(label='Number of characters to generate',
                               min_value=1, max_value=5000, value=500,
                               format='%d')

# Name of the character that initiates the text
character_option = st.selectbox('Choose the main character',
                                ('Harry', 'Hermione', 'Ron',
                                 'Dumbledore', 'McGonagall', 'Voldemort'))

if character_option == 'Voldemort':
    st.markdown(
        '`I see that you are not afraid of him. You look brave, are you a Griffindor? or...`')

# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
temperature = st.slider(label='Tempeture (low: predictable, high: weird)', 
                        min_value=0.01, max_value=1.5, value=0.8)



path_to_file = 'data/hp1.txt'
# Read text
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# Unique characters in the text
vocab = sorted(set(text))

# Map each unique char to a different index
char2idx = {u: i for i, u in enumerate(vocab)}
# Map the index to the respective char
idx2char = np.array(vocab)

if st.button('Create Story'):
    # Length of the vocabulary in charts
    vocab_size = len(vocab)
    # The embedding dimension
    embedding_dim = 256
    # Number of RNN units
    rnn_units = 1024
    # Build network structure
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Load the weights of our latest learned model
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    # Build the learned model
    model.build(tf.TensorShape([1, None]))

    # Make predictions
    predicted_text = generate_text(
        model, start_string=character_option + ' ', num_generate=int(num_generate), temperature=temperature)

    st.write(predicted_text)

st.text('')
st.markdown(
    '`Create by` [santiviquez](https://twitter.com/santiviquez) |  `Code:` [GitHub](https://github.com/santiviquez/harry-potter-rnn/)')
