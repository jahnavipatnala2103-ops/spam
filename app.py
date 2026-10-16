"""
AI Spam Message Detector
------------------------
A minimal Streamlit app that classifies a text message as SPAM or NOT SPAM.

Method: TF-IDF (turns text into numbers based on word importance)
        + Multinomial Naive Bayes (a simple, fast text classifier).

IMPORTANT: By default the model is trained on a SMALL DEMONSTRATION dataset
(data/spam_demo_dataset.csv, ~80 hand-written messages). It is a learning demo,
NOT a production-grade spam filter, and it will make mistakes.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Path to the labeled dataset (columns: label, text; label is "spam" or "ham").
# To use a larger dataset, replace this file with one using the same columns.
DATASET_PATH = Path(__file__).parent / "data" / "spam_demo_dataset.csv"


@st.cache_resource  # Train once and reuse, instead of retraining on every click
def load_model():
    """Load the dataset and train the TF-IDF + Naive Bayes pipeline."""
    df = pd.read_csv(DATASET_PATH)

    # Basic cleaning: drop empty rows and normalise labels
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both 'spam' and 'ham' examples.")

    y = (df["label"] == "spam").astype(int)  # 1 = spam, 0 = not spam

    model = make_pipeline(
        # Lowercase, remove common English words, use single words + word pairs
        TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2)),
        MultinomialNB(alpha=0.5),  # alpha smooths rare/unseen words
    )
    model.fit(df["text"], y)
    return model, len(df)


# ---------------------------- Page layout ----------------------------
st.set_page_config(page_title="Spam Message Detector", page_icon="🛡️")

st.title("🛡️ Spam Message Detector")
st.write(
    "Paste a message below and click **Check Message** to see whether it "
    "looks like spam."
)

# Try to load the model; show a friendly error instead of crashing
try:
    model, n_examples = load_model()
except FileNotFoundError:
    st.error(f"Dataset not found. Expected file at: `{DATASET_PATH}`")
    st.stop()
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

with st.expander("ℹ️ About this detector (please read)"):
    st.write(
        f"This app uses TF-IDF + Naive Bayes trained on a **small demonstration "
        f"dataset of {n_examples} example messages**. It is meant for learning, "
        "not for real-world filtering. It can miss spam and can wrongly flag "
        "genuine messages. Always use your own judgment, and never click "
        "suspicious links or share passwords or card details."
    )

message = st.text_area(
    "Message to check", height=160, placeholder="Type or paste a message here..."
)

# ---------------------------- Detection ----------------------------
if st.button("Check Message", type="primary"):
    # Empty-input validation (also catches whitespace-only input)
    if not message.strip():
        st.error("Please enter a message before clicking 'Check Message'.")
    else:
        try:
            is_spam = model.predict([message])[0] == 1
            spam_prob = model.predict_proba([message])[0][1]  # model's estimate

            st.subheader("Result")
            if is_spam:
                st.warning(
                    "🚨 **SPAM** — This message looks like spam. Be careful with "
                    "any links, attachments, or requests for personal information."
                )
            else:
                st.info(
                    "✅ **NOT SPAM** — No obvious spam indicators were detected. "
                    "This is not a guarantee that the message is safe."
                )
            st.caption(
                f"Model's spam score: {spam_prob:.0%} (a rough estimate from a "
                "small demo model, not a certainty)."
            )
        except Exception as e:
            st.error(f"Something went wrong while analysing the message: {e}")
