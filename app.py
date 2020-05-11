import streamlit as st
from model.predict_harry_potter import generate_text, load_model, calculate_id_char_mapping
import os
import tensorflow as tf
import typing

def make_predictions(model, character_option, char2idx,idx2char, num_generate, temperature):
    # Make predictions
    predicted_text = generate_text(
        model, start_string=character_option + ' ',char2idx=char2idx,idx2char=idx2char, num_generate=int(num_generate), temperature=temperature)
    return predicted_text


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

@st.cache(allow_output_mutation=True, hash_funcs=None)
def load_tf_model():
    print("calculating load_tf_model. SHOULD ONLY BECALLED ONCE")
    return load_model()

@st.cache()
def build_char_idx():
    print("calculating char2idx. SHOULD ONLY BECALLED ONCE")
    return calculate_id_char_mapping()

if st.button('Create Story'):
        print("\nRunning new story request")
        char2idx, idx2char = build_char_idx()
        model = load_tf_model()
        predicted_text = make_predictions(
            model, character_option,char2idx,idx2char, num_generate, temperature)
        st.write(predicted_text)


st.text('')
st.markdown(
    '`Create by` [santiviquez](https://twitter.com/santiviquez) |  `Code:` [GitHub](https://github.com/santiviquez/harry-potter-rnn/)')
