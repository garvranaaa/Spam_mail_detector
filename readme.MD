# 🛡️ Spam Detector

NLP classifier that detects spam messages with a probability score. Built with Scikit-learn and Streamlit, trained on 5,574 real SMS messages.

🔗 **Live Demo:** *(add Streamlit Cloud URL after deploying)*

---

## Demo

![App Demo](app_demo.png)

---

## Features

**Probability Score — not just a binary label**
Every message gets a spam probability (0–100%) with a visual confidence bar. Borderline cases are flagged differently from high-confidence spam.

**Full NLP Pipeline**
Text goes through regex cleaning → lowercasing → NLTK stopword removal → Porter stemming before TF-IDF vectorization. Same pipeline at train time and inference time — no leakage.

**Model Performance Dashboard**
Confusion matrix with TN/TP/FP/FN breakdown, precision/recall/F1 metrics with explanations, top spam-indicating words by model coefficient, and class distribution chart.

**One-click Examples**
Three pre-loaded examples (two spam, one ham) for instant testing.

---

## Model

| | |
|---|---|
| Dataset | SMS Spam Collection — 5,574 messages |
| Features | TF-IDF, 5,000 features, bigrams `(1,2)-ngrams` |
| Classifier | Logistic Regression (C=5, L2 regularization) |
| Split | 80/20, stratified by class |

### Results

| Metric | Score |
|---|---|
| Accuracy | ~98% |
| Precision | ~97% |
| Recall | ~93% |
| F1 | ~95% |

---

## Preprocessing Pipeline

```
Raw text
  → Strip special characters (regex)
  → Lowercase
  → Remove NLTK English stopwords ("the", "is", "at"...)
  → Porter stemming ("winning" → "win", "free!" → "free")
  → TF-IDF vectorization (5,000 features, unigrams + bigrams)
```

TF-IDF rewards words that appear often in *this* message but rarely across *all* messages — so "free" in a spam message scores higher than "the" which is everywhere.

---

## Why Logistic Regression over Naive Bayes

Naive Bayes assumes feature independence — every word is treated as unrelated to every other. Logistic Regression learns feature correlations directly, handles bigrams better, and outputs better-calibrated probability scores. For a spam detector where confidence level matters (not just the binary label), LR is the stronger choice.

---

## Why Precision Matters More Than Recall Here

A false positive (legitimate email flagged as spam) is more disruptive than a false negative (spam that gets through). High precision minimises false alarms. The model is tuned accordingly.

---

## Run Locally

```bash
git clone https://github.com/garvranaaa/spam-detector
cd spam-detector

python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
source .venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

Make sure `spam.csv` is in the same folder as `app.py`.  
Dataset: [SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## Project Structure

```
spam-detector/
├── app.py            # Streamlit app
├── spam.csv          # SMS Spam Collection dataset
├── requirements.txt
└── README.md
```

---

## Stack

| Library | Purpose |
|---|---|
| Scikit-learn | TF-IDF vectorizer, Logistic Regression, metrics |
| NLTK | Stopword removal, Porter stemmer |
| Streamlit | Web app |
| Matplotlib | Confusion matrix, coefficient chart |
| Pandas | Data loading and preprocessing |

---

Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) · [GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)