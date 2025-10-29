import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("spam_or_notSpam.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
max_len = 100

# Streamlit App UI
st.set_page_config(page_title="📧 Spam Mail Detection", page_icon="🤖", layout="centered")
st.title("📩 Spam Mail Detection System")
st.write("Built with **TensorFlow, Streamlit, and NLP**")

# Input Box
user_input = st.text_area("✍️ Enter the email/message text below:", height=150)

if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message before submitting.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        pad = pad_sequences(seq, maxlen=max_len)
        pred = model.predict(pad)[0][0]
        label = "🚨 Spam" if pred > 0.5 else "✅ Not Spam"
        st.subheader(label)
        st.caption(f"Prediction Confidence: {pred:.4f}")

st.markdown("---")
st.markdown("👨‍💻 *Developed by Adithya  — AI & Deep Learning Project*")
