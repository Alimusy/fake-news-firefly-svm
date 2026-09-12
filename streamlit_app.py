import streamlit as st

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.subheader("Binary Firefly Algorithm + Support Vector Machine")

st.markdown("""
This system detects fake news articles using two models:

- **Baseline SVM** — trained on all 5,000 TF-IDF features
- **Firefly + SVM** — trained on 3,594 optimally selected features
  using the Binary Firefly Algorithm (28.1% dimensionality reduction)

---

### How to use
Use the sidebar to navigate between the two models.
Paste any news article into the text box and click **Predict**.

---

### Dataset
Trained on the **WELFake dataset** — 72,095 news articles
from four sources, balanced between real and fake news.

---

### Model Performance

| Model | Accuracy | F1 Score | Features |
|---|---|---|---|
| Baseline SVM | 95.40% | 95.55% | 5,000 |
| Firefly + SVM | 94.83% | 94.99% | 3,594 |
""")

st.info("👈 Select a model from the sidebar to get started.")
