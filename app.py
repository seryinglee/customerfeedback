import streamlit as st
import pandas as pd
import datetime
import time
from typing import List
from model import train_logreg_model, train_svm_model, train_bert_model
from reply import generate_reply

if "user_counter" not in st.session_state:
    st.session_state.user_counter = 1

st.set_page_config(
    page_title="Review Auto Reply System",
    page_icon="💬",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.main {
    background: linear-gradient(180deg, #f0f6ff, #ffffff);
    padding: 0;
}
h1 {
    color: #1976D2;
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}
.css-1d391kg, .css-1vq4p4l {
    background-color: #e3f2fd !important;
    border-right: 1px solid #bbdefb;
}
.stButton>button {
    background: linear-gradient(90deg, #1976D2, #2196F3);
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1565C0, #1976D2);
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
textarea {
    border: 1.5px solid #90caf9 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    padding: 10px !important;
    font-size: 14px !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5faff, #ffffff);
    border-right: 1px solid #d6e4f0;
    padding: 24px 20px !important;
}
.sidebar-title {
    font-size: 17px;
    font-weight: 600;
    color: #0d47a1;
    margin: 18px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #1976d2;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
model_choice = st.radio("Select Model", ["SVM", "Logistic Regression", "BERT"])

@st.cache_resource
def _load_simple_model(choice):
    if choice == "Logistic Regression":
        model, _ = train_logreg_model()
    elif choice == "SVM":
        model, _ = train_svm_model()
    return model

if "bert_model" not in st.session_state:
    st.session_state.bert_model = None

if model_choice in ["Logistic Regression", "SVM"]:
    with st.spinner(f'🤖 Loading {model_choice} model...'):
        model = _load_simple_model(model_choice)
elif model_choice == "BERT":
    if st.session_state.bert_model is None:
        st.warning("⚠️ BERT needs to be trained. This will take some time.")
        if st.button("Start BERT Training"):
            with st.spinner('🦾 Training BERT...'):
                model, _ = train_bert_model()
                st.session_state.bert_model = model
            st.success("BERT trained successfully!")
        else:
            st.stop()
    else:
        model = st.session_state.bert_model

if model is None:
    st.error("Model could not be loaded.")
    st.stop()

st.title("💬 Review Auto Reply System")
st.markdown("Predict review type (Positive / Negative / Neutral / Critical) and generate auto-replies.")

tab1, tab2 = st.tabs(["💬 Single Review", "📦 Batch Prediction"])

with tab1:
    st.markdown("### 🔖 Quick Examples")
    examples = [
        "This product is amazing, I love it so much!",
        "Terrible quality, very bad.",
        "It’s okay, nothing special.",
        "The packaging was damaged when I received it.",
    ]
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["user_review"] = ex

    st.subheader("Try a Review")
    user_review = st.text_area("Enter a review text", key="user_review", placeholder="Type or click a quick example above...", height=120)

    if st.button("🔍 Predict & Reply", type="primary", key="predict_btn"):
        if not user_review.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("🤖 Analyzing your review..."):
                final_label = model.predict([user_review])[0]
                auto_reply = generate_reply(final_label)

            label_colors = {
                "Positive": "#2E7D32",
                "Negative": "#C62828",
                "Neutral": "#616161",
                "Critical": "#EF6C00"
            }
            label_color = label_colors.get(final_label, "#1565C0")

            username = f"Customer_{st.session_state.user_counter}"
            st.session_state.user_counter += 1
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            star_rating = "⭐️⭐️⭐️⭐️⭐️" if final_label=="Positive" else "⭐️⭐️" if final_label=="Negative" else "⭐️⭐️⭐️"

            st.markdown(f"""
            <div style="border:1px solid #e0e0e0;border-radius:18px;padding:20px;
                 margin:20px 0 25px 0;background:white;
                 box-shadow:0 4px 12px rgba(0,0,0,0.08);font-family:'Segoe UI',sans-serif;">
                <div style="display:flex; align-items:center; margin-bottom:12px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png" width="46" 
                         style="border-radius:50%; margin-right:12px;"/>
                    <div>
                        <b style="font-size:16px; color:#222;">{username}</b><br>
                        <span style="font-size:12px; color:#888;">{timestamp}</span>
                    </div>
                    <span style="margin-left:auto; padding:6px 14px; border-radius:999px; 
                                 background:{label_color}; color:white; font-size:13px; font-weight:bold;">
                        {final_label}
                    </span>
                </div>
                <div style="margin-left:58px; font-size:14px; color:#f39c12; margin-bottom:8px;">
                    {star_rating}
                </div>
                <div style="margin-left:58px; font-size:15px; line-height:1.6; color:#333;">
                    {user_review}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Official Reply")
            placeholder = st.empty()
            typed_text = ""
            for char in auto_reply:
                typed_text += char
                placeholder.info(typed_text)
                time.sleep(0.03)

with tab2:
    st.subheader("Upload a CSV for batch predictions")
    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_uploader")

    if batch_file is not None:
        try:
            df_batch = pd.read_csv(batch_file)
            if "text" not in df_batch.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                preds, replies = [], []
                for t in df_batch["text"].fillna(""):
                    label = model.predict([str(t)])[0]
                    preds.append(label)
                    replies.append(generate_reply(label))
                df_batch["predicted"] = preds
                df_batch["auto_reply"] = replies

                for i, row in df_batch.iterrows():
                    st.markdown(f"""
                    <div style="border:1px solid #e0e0e0;border-radius:16px;padding:15px;
                         margin:15px 0;background:#fff;box-shadow:0 2px 6px rgba(0,0,0,0.06);">
                        <b style="color:#1976D2;">Review:</b> {row['text']}<br>
                        <b style="color:#C62828;">Predicted:</b> {row['predicted']}<br>
                        <b style="color:#2E7D32;">Reply:</b> {row['auto_reply']}
                    </div>
                    """, unsafe_allow_html=True)

                csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results CSV",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
