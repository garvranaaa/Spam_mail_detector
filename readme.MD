# 🛡️ Spam Detector

A machine learning-powered SMS spam detector that classifies messages with confidence scores. Built with **Scikit-learn**, **NLTK**, and **Streamlit**, trained on 5,574+ real SMS messages from the UCI ML repository.

**Detects spam signals** including urgency tactics, false claims, financial offers, and overpromises — with detailed breakdowns showing exactly why a message looks suspicious.

🔗 **Live Demo:** [spam-mail-det.streamlit.app](https://spam-mail-det.streamlit.app/)

---

## ✨ Features

- **Probability Scores** — Not just binary classification. Every message gets a spam confidence score (0–100%) with visual indicators
- **Signal Categories** — Highlights 4 types of spam signals: Urgency, Shady, Financial, and Overpromise language
- **Full NLP Pipeline** — Regex cleaning → lowercasing → stopword removal → Porter stemming → TF-IDF vectorization
- **Performance Dashboard** — Confusion matrix, precision/recall/F1 metrics, top spam-indicating words, dataset distribution
- **Interactive Examples** — One-click examples for instant testing (prize scams, investment schemes, normal text)
- **Beautiful UI** — Clean, responsive design with custom Streamlit styling

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/garvranaaa/Spam_mail_detector.git
cd Spam_mail_detector

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
source .venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run spam_app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~98% |
| **Precision** | ~97% |
| **Recall** | ~93% |
| **F1 Score** | ~95% |

**Why these metrics matter:**
- **Precision** is prioritized to minimize false alarms (legitimate emails flagged as spam)
- **Recall** is secondary because missing spam is less disruptive than blocking legitimate messages

---

## 🔧 Model Details

| Component | Details |
|-----------|---------|
| **Dataset** | SMS Spam Collection (UCI ML) — 5,574 messages (4,827 ham, 747 spam) |
| **Classifier** | Logistic Regression (C=5, L2 regularization) |
| **Features** | TF-IDF vectorization with 5,000 features + bigrams (1,2)-ngrams |
| **Train/Test Split** | 80/20 with stratification |

### NLP Pipeline

```
Raw Message
    ↓
Strip special characters (regex \W)
    ↓
Lowercase
    ↓
Remove NLTK English stopwords
    ↓
Porter stemming (winning → win, guaranteed → guaranty)
    ↓
TF-IDF Vectorization (5,000 features)
    ↓
Logistic Regression Classifier
    ↓
Spam Probability Score (0–100%)
```
├── app.py            # Streamlit app
├── spam.csv          # SMS Spam Collection dataset
├── requirements.txt
└── README.md
```

---

## 🎯 Signal Categories

The app detects 4 types of spam indicators:

| Category | Pattern | Example |
|----------|---------|---------|
| **🚨 Urgency** | urgent, asap, deadline, act now | "Reply NOW — limited time offer!" |
| **🕵️ Shady** | guaranteed, winner, claim, congratulations | "You've been selected as our winner!" |
| **💰 Financial** | free, cash, money, profit, bonus | "Earn FREE cash instantly" |
| **🤯 Overpromise** | million, billion, risk-free, no cost | "Make $1M working from home!" |

---

## 🧠 Model Architecture

**Why Logistic Regression?**
- Learns feature correlations directly (unlike Naive Bayes which assumes independence)
- Provides well-calibrated probability scores (not just binary labels)
- Handles bigrams effectively
- Fast training and inference
- Interpretable coefficients show which words signal spam

**Why TF-IDF?**
- Rewards words that appear often in *this* message but rarely across *all* messages
- For example: "free" in a spam message gets higher weight than "the" (which appears everywhere)

---

## 📁 Project Structure

```
spam-detector/
├── spam_app.py              # Main Streamlit app
├── spam.csv                 # Dataset (5,574 messages)
├── requirements.txt         # Python dependencies
├── readme.MD                # This file
└── tmp_check.py             # Testing utilities
```

---

## 📦 Dependencies

```
streamlit>=1.0
pandas>=1.3
scikit-learn>=1.0
nltk>=3.6
matplotlib>=3.5
```

---

## 🔍 How to Use

1. **Paste a message** — Copy any SMS, email, or message text
2. **Click "Check for spam"** — The model analyzes it in real-time
3. **View results** — See the probability score, detected signals, and category breakdown
4. **Explore metrics** — Switch to the "Performance" tab to see model accuracy and confusion matrix

---

## 📚 Dataset Source

**SMS Spam Collection** from [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,574 real SMS messages
- Binary classification: ham vs spam
- Imbalanced dataset (13.4% spam, 86.6% ham)

---

## 💡 Key Insights

- **High Precision (97%)** — Only 3% of flagged messages are actually legitimate
- **Strong Recall (93%)** — Catches 93% of actual spam
- **Trade-off:** Model prioritizes precision to minimize false alarms
- **Feature Importance:** Words like "free", "winner", "urgent", "claim" are top spam indicators

---

## 🚀 Future Enhancements

- [ ] Deploy on Streamlit Cloud
- [ ] Add email attachment analysis
- [ ] Multi-language support
- [ ] User feedback loop to retrain model
- [ ] Integration with email providers (Gmail, Outlook)

---

## 📄 License

This project is open source and available under the MIT License.

---

**Built by:** Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in)

**Connect:** [GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
