import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("spam_or_notSpam.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100  # must match training

# Example emails
mails = [
    "Win a brand new car! Click this link now.",
    "Hi friend, are we still meeting tomorrow?",
    "You have been selected for a $1000 gift card."
]

# Convert to sequences
mail_seq = tokenizer.texts_to_sequences(mails)
mail_pad = pad_sequences(mail_seq, maxlen=max_len)

# Predict
preds = model.predict(mail_pad)

for i, pred in enumerate(preds):
    label = "Spam" if pred[0] > 0.5 else "Not Spam"
    print(f"📧 Mail: {mails[i]}")
    print(f"   Prediction: {label} (Score: {pred[0]:.4f})\n")
