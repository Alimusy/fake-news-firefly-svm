import streamlit as st
import pickle
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

st.set_page_config(page_title="Firefly + SVM", page_icon="✨", layout="centered")

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base, 'models')
    with open(os.path.join(models_dir, 'fa_svm.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(models_dir, 'bfa_best_position.pkl'), 'rb') as f:
        best_position = pickle.load(f)
    selected_indices = np.where(best_position == 1)[0]
    return model, tfidf, selected_indices

model, tfidf, selected_indices = load_models()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def get_confidence(model, vec):
    decision = model.decision_function(vec)[0]
    return round((1 / (1 + np.exp(-abs(decision)))) * 100, 2)

st.title("✨ Firefly + SVM Prediction")
st.markdown("Predicts using **3,594 optimally selected features** via Binary Firefly Algorithm.")
st.markdown("**Model Accuracy: 94.83% | F1 Score: 94.99% | Feature Reduction: 28.1%**")
st.divider()

article = st.text_area("Paste your news article here:", height=250,
                       placeholder="Enter the full text of the news article...")

if st.button("🔍 Predict", use_container_width=True):
    if not article.strip():
        st.warning("Please enter an article to analyse.")
    else:
        with st.spinner("Analysing..."):
            clean  = preprocess(article)
            vec    = tfidf.transform([clean]).toarray()
            vec_fa = vec[:, selected_indices]
            pred   = model.predict(vec_fa)[0]
            conf   = get_confidence(model, vec_fa)

        st.divider()
        # NOTE: 0 = Real, 1 = Fake
        if pred == 1:
            st.error("🚨 FAKE NEWS DETECTED")
            st.markdown("""
            <div style='background-color:#fff0f0; padding:20px; border-radius:10px;
                        border-left:5px solid #ff4b4b;'>
                <h3 style='color:#ff4b4b; margin:0'>FAKE</h3>
                <p style='color:#333; margin:5px 0 0 0'>This article is likely <b>fake</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ REAL NEWS DETECTED")
            st.markdown("""
            <div style='background-color:#f0fff4; padding:20px; border-radius:10px;
                        border-left:5px solid #21c55d;'>
                <h3 style='color:#21c55d; margin:0'>REAL</h3>
                <p style='color:#333; margin:5px 0 0 0'>This article appears to be <b>genuine</b>.</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.metric(label="Confidence Score", value=f"{conf}%")
        st.progress(int(conf))
        st.divider()
        st.markdown("#### Article Summary")
        st.markdown(f"- **Words after preprocessing:** {len(clean.split())}")
        st.markdown(f"- **Features used:** {len(selected_indices)} / 5,000")
        st.markdown(f"- **Feature reduction:** 28.1%")
        st.markdown(f"- **Model:** Linear SVM + Binary Firefly Algorithm")