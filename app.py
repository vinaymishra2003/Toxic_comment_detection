import streamlit as st
import pandas as pd
from model import load_model, predict

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="🛑",
    layout="centered"
)

# ---------------- Header ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🛑 Toxic Comment Detection</h1>
    <p style="text-align:center; font-size:16px;">
    BERT-based multi-label classification system
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- About ----------------
with st.expander("📌 About This Application", expanded=True):
    st.markdown(
        """
        **Purpose**  
        This application detects harmful and abusive language in text using a 
        **fine-tuned BERT deep learning model**.

        **Why this matters**  
        Toxic comments can promote harassment, hate speech, and online abuse.  
        This system helps automate **content moderation** for platforms and communities.

        **Toxicity Categories**
        - Toxic  
        - Severe Toxic  
        - Obscene  
        - Threat  
        - Insult  
        - Identity Hate  
        """
    )

# ---------------- Load Model ----------------
@st.cache_resource
def load_resources():
    return load_model()

model, tokenizer = load_resources()

# ---------------- Single Comment ----------------
st.subheader("🔍 Analyze a Comment")

text = st.text_area(
    "Enter a comment",
    placeholder="Example: You are absolutely useless..."
)

if st.button("Analyze"):
    if text.strip():
        scores, flags = predict(text, model, tokenizer)

        st.subheader("📊 Toxicity Scores")
        st.json(scores)

        if any(flags.values()):
            st.error("⚠️ Toxic content detected")
        else:
            st.success("✅ No significant toxicity detected")
    else:
        st.warning("Please enter some text.")

st.divider()

# ---------------- Label Explanation ----------------
with st.expander("📖 Understanding the Labels"):
    st.markdown(
        """
        - **toxic**: General abusive language  
        - **severe_toxic**: Extremely aggressive content  
        - **obscene**: Profanity or vulgar expressions  
        - **threat**: Threats of violence or harm  
        - **insult**: Personal attacks  
        - **identity_hate**: Hate speech targeting groups
        """
    )

# ---------------- Bulk CSV ----------------
st.subheader("📁 Bulk CSV Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file (must contain `comment_text` column)",
    type="csv"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "comment_text" not in df.columns:
        st.error("❌ CSV must contain a `comment_text` column")
    else:
        results = []
        for text in df["comment_text"].astype(str):
            scores, _ = predict(text, model, tokenizer)
            results.append(scores)

        result_df = pd.concat([df, pd.DataFrame(results)], axis=1)

        st.success("✅ Prediction completed")
        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            "⬇ Download Results",
            result_df.to_csv(index=False),
            file_name="toxicity_predictions.csv",
            mime="text/csv"
        )

# ---------------- Footer ----------------
st.divider()
st.markdown(
    """
    <p style="text-align:center; font-size:13px; color:gray;">
    Built using BERT, PyTorch, Hugging Face & Streamlit<br>
    Toxic Comment Classification System
    </p>
    """,
    unsafe_allow_html=True
)
