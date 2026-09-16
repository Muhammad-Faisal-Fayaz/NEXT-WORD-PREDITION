from pathlib import Path
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "next_word_lstm.keras"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.pkl"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            "The trained model is missing or empty. Run the final training cell in project.ipynb first."
        )
    if not TOKENIZER_PATH.exists() or TOKENIZER_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            "The tokenizer is missing or empty. Run the final training cell in project.ipynb first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


def predict_next_words(seed_text, model, tokenizer, top_k):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if not token_list:
        return []

    sequence_length = model.input_shape[1]
    model_input = pad_sequences(
        [token_list],
        maxlen=sequence_length,
        padding="pre",
        truncating="pre",
    )
    probabilities = model.predict(model_input, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    index_word = {index: word for word, index in tokenizer.word_index.items()}

    return [
        (index_word[index], float(probabilities[index]))
        for index in top_indices
        if index in index_word
    ]


st.set_page_config(page_title="Next Word Predictor", page_icon="✍️")
st.title("Next Word Predictor")
st.write("Type a phrase and let the trained LSTM suggest what comes next.")

try:
    model, tokenizer = load_artifacts()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

seed_text = st.text_area(
    "Enter a phrase",
    value="the",
    height=120,
    placeholder="Start typing here...",
)
top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)

if st.button("Predict next words", type="primary"):
    if not seed_text.strip():
        st.warning("Enter at least one word.")
    else:
        suggestions = predict_next_words(seed_text, model, tokenizer, top_k)
        if not suggestions:
            st.warning("No known words were found in that phrase.")
        else:
            st.subheader("Suggestions")
            for word, probability in suggestions:
                st.write(f"**{word}** - {probability:.1%}")
