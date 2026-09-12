# Fake News Detection with Binary Firefly Feature Selection and SVM

Fake news classifier that compares a standard SVM trained on the full TF-IDF
feature space against an SVM trained on a feature subset chosen by a Binary
Firefly Algorithm (BFA).

## Open it

Training and evaluation notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alimusy/fake-news-firefly-svm/blob/main/notebooks/fake_news_firefly_svm.ipynb)

Deploy the app yourself, one click, free:

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Alimusy/fake-news-firefly-svm&branch=main&mainModule=app/app.py)

## Method

1. Article text cleaned and vectorised into 5,000 TF-IDF features.
2. A Binary Firefly Algorithm searches the feature space, encoding each
   candidate subset as a binary vector and scoring it with the classification
   accuracy of an internal SVM.
3. Two SVMs are trained under identical settings, one on all features and one
   on the BFA subset, so the only difference is feature selection.

## Results

| Model | Features | Accuracy | F1 |
|---|---|---|---|
| Baseline SVM | 5,000 | 95.40% | 95.55% |
| BFA + SVM | 3,594 | 94.83% | 94.99% |

A 28.1% cut in dimensionality costs about half a percentage point of accuracy.

## Repo layout

```
notebooks/   training and evaluation notebook
app/         Streamlit app (app.py + pages/)
app/app/models/  trained SVMs, TF-IDF vectoriser, selected feature mask
```

## Running the app

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Dataset

WELFake fake news dataset. Not committed here; download it and point the
notebook at your local copy.
