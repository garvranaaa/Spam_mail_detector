import re
import html as html_lib
import warnings
import logging
from pathlib import Path
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"

st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="wide")

# ── Category definitions ───────────────────────────────────────────────────────

CATEGORIES = {
    "Urgency": {
        "pattern": r'\b(urgent|asap|immediately|hurry|expires?|today only|act now|deadline|last chance|right now|reply now|respond)\b',
        "bg": "#FEE2E2", "border": "#EF4444", "text": "#991B1B", "emoji": "🚨"
    },
    "Shady": {
        "pattern": r'\b(guaranteed?|winner|prize|won|claim|selected|congratulations?|chosen|exclusive|secret|confidential|private)\b',
        "bg": "#FEF9C3", "border": "#EAB308", "text": "#854D0E", "emoji": "🕵️"
    },
    "Financial": {
        "pattern": r'\b(free|cash|money|loan|credit|investment?|profit|earn|reward|gift|bonus|discount|fund|finance)\b',
        "bg": "#EDE9FE", "border": "#8B5CF6", "text": "#5B21B6", "emoji": "💰"
    },
    "Overpromise": {
        "pattern": r'\b(million|billion|risk.?free|no cost|no obligation|unlimited|lifetime|forever|100 percent|no strings)\b',
        "bg": "#FCE7F3", "border": "#EC4899", "text": "#9D174D", "emoji": "🤯"
    },
}

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700&family=DM+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background-color: #FFFFFF !important;
    color: #1C1C1A !important;
    font-family: 'DM Sans', sans-serif !important;
}

* { box-shadow: none !important; text-shadow: none !important; }

[data-testid="stSidebar"]    { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }

h1, h2, h3, h4 { color: #1C1C1A !important; font-weight: 700 !important; }

/* Text area */
textarea {
    background-color: #FAFAF8 !important;
    border: 1.5px solid #D4D9BE !important;
    border-radius: 8px !important;
    color: #1C1C1A !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.75 !important;
}
textarea:focus { border-color: #4A5A2A !important; outline: none !important; }

/* Primary button */
.stButton > button[kind="primary"] {
    background-color: #4A5A2A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"]:hover { background-color: #3A4820 !important; }

/* Secondary buttons */
.stButton > button {
    background-color: #F7F7F2 !important;
    color: #4A5A2A !important;
    border: 1.5px solid #D4D9BE !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stButton > button:hover { background-color: #ECEEE0 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #F7F7F2 !important;
    border: 1.5px solid #E4E8D4 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: #4A5A2A !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] { color: #6B6B65 !important; font-size: 0.82rem !important; }

hr { border: none !important; border-top: 1.5px solid #E4E8D4 !important; margin: 1.2rem 0 !important; }
.stCaption, small { color: #6B6B65 !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px !important; border-left: 4px solid !important; }
.stSuccess { background-color: #F0F4E8 !important; border-color: #4A5A2A !important; }
.stSuccess * { color: #2A3A12 !important; }
.stError   { background-color: #FAF0EE !important; border-color: #B85C3A !important; }
.stError *   { color: #7A2A10 !important; }
.stWarning { background-color: #FDF6EC !important; border-color: #C4882A !important; }
.stWarning * { color: #7A4A10 !important; }
.stInfo    { background-color: #F4F4EE !important; border-color: #6B7A3E !important; }
.stInfo *    { color: #2A3A12 !important; }

/* Tab strip */
[data-baseweb="tab-list"] {
    background: #F7F7F2 !important;
    border-bottom: 2px solid #D4D9BE !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #6B6B65 !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}
[aria-selected="true"] {
    background: #FFFFFF !important;
    color: #4A5A2A !important;
    border-bottom: 2px solid #4A5A2A !important;
}

/* Chart area */
[data-testid="stImage"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── NLTK ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english")), PorterStemmer()

stop_words, stemmer = setup_nltk()

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def train_model():
    if not DATA_PATH.exists():
        return None, None, None, None, "spam.csv not found.", None
    try:
        df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=["v1", "v2"])
    except Exception as e:
        return None, None, None, None, str(e), None

    df.columns = ["label", "message"]
    df = df.dropna()
    df["label"]   = df["label"].map({"ham": 0, "spam": 1})
    df = df[df["label"].isin([0, 1])]
    df["cleaned"] = df["message"].apply(preprocess)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["cleaned"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, C=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "cm":        confusion_matrix(y_test, y_pred),
    }

    feat   = vectorizer.get_feature_names_out()
    coefs  = model.coef_[0]
    top_spam = [(feat[i], float(coefs[i])) for i in coefs.argsort()[-15:][::-1]]

    return model, vectorizer, metrics, top_spam, None, df

def classify(text, model, vectorizer):
    vec  = vectorizer.transform([preprocess(text)])
    prob = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    return int(pred), float(prob[1])

# ── Category detection + highlighting ─────────────────────────────────────────

def detect_categories(text):
    """Return {category: [matched_words]} for found patterns."""
    found = {}
    for cat, cfg in CATEGORIES.items():
        matches = re.findall(cfg["pattern"], text, re.IGNORECASE)
        if matches:
            found[cat] = matches
    return found

def highlight_html(text, categories_found):
    """Return safe HTML with category-colored highlights."""
    # Build list of (start, end, category) for all matches
    spans = []
    for cat, cfg in CATEGORIES.items():
        for m in re.finditer(cfg["pattern"], text, re.IGNORECASE):
            spans.append((m.start(), m.end(), cat))

    # Sort and de-overlap
    spans.sort(key=lambda x: x[0])
    clean_spans = []
    last_end = 0
    for s, e, cat in spans:
        if s >= last_end:
            clean_spans.append((s, e, cat))
            last_end = e

    # Build HTML
    parts = []
    pos = 0
    for s, e, cat in clean_spans:
        parts.append(html_lib.escape(text[pos:s]))
        cfg = CATEGORIES[cat]
        parts.append(
            f'<mark style="background:{cfg["bg"]};border-bottom:2px solid {cfg["border"]};'
            f'color:{cfg["text"]};padding:1px 3px;border-radius:3px;font-weight:500;">'
            f'{html_lib.escape(text[s:e])}</mark>'
        )
        pos = e
    parts.append(html_lib.escape(text[pos:]))
    return ''.join(parts)

def score_label(spam_prob):
    if spam_prob < 0.2:  return "Safe",      "#16A34A"
    if spam_prob < 0.5:  return "Suspicious", "#D97706"
    if spam_prob < 0.75: return "Poor",       "#EA580C"
    return                      "Dangerous",   "#DC2626"

def word_count(text): return len(text.split())
def read_time(text):
    wc = word_count(text)
    if wc < 50:  return "a few seconds"
    if wc < 200: return f"~{max(1, wc // 200)} min"
    return f"~{wc // 200} min"

# ── Load model ────────────────────────────────────────────────────────────────

with st.spinner("Loading model..."):
    model, vectorizer, metrics, top_spam, err, df = train_model()

if err:
    st.error(err)
    st.stop()

# Narrow types for static analysis (Pylance/Pyright)
assert model is not None
assert vectorizer is not None
assert metrics is not None
assert top_spam is not None
assert df is not None

# ── Session state ─────────────────────────────────────────────────────────────

for k, v in {"view": "input", "msg": "", "pred": None, "spam_prob": 0.0,
             "categories": {}, "highlighted": ""}.items():
    st.session_state.setdefault(k, v)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding:2.5rem 0 0.5rem 0;">
  <h1 style="font-size:3rem; letter-spacing:-1px; margin-bottom:0;">🛡️ Spam Detector</h1>
  <p style="font-size:1.05rem; color:#6B6B65; margin-top:0.4rem;">
    Paste any message to detect spam signals, word categories, and get a confidence score.
  </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ── Main layout: Analyse tab + Performance tab ────────────────────────────────

main_tab, perf_tab, about_tab = st.tabs(["🔍 Analyse", "📊 Performance", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════
# ANALYSE TAB
# ═══════════════════════════════════════════════════════════
with main_tab:

    # Example strip
    st.caption("Try an example:")
    ex1, ex2, ex3, _ = st.columns([1, 1, 1, 3])
    if ex1.button("🚨 Prize scam"):
        st.session_state.msg  = "Congratulations! You've won a £1,000 Tesco gift card. Go to www.claim-prize.com to claim your free reward NOW. Limited time only — ASAP!"
        st.session_state.view = "input"
    if ex2.button("🚨 Investment"):
        st.session_state.msg  = "Dear friend, I am a Financial Consultant in control of privately owned funds placed for long term investments. Guaranteed 5% ROI. Please answer ASAP."
        st.session_state.view = "input"
    if ex3.button("✅ Normal text"):
        st.session_state.msg  = "Hey! Are we still on for lunch tomorrow? Let me know what time works best for you."
        st.session_state.view = "input"

    st.write("")

    # ── INPUT VIEW ──────────────────────────────────────────
    if st.session_state.view == "input":
        user_input = st.text_area(
            "Message",
            value=st.session_state.msg,
            height=280,
            placeholder="Copy/paste an email or SMS message to check for spam...",
            label_visibility="collapsed"
        )
        st.write("")
        if st.button("Check for spam →", type="primary"):
            if not user_input.strip():
                st.warning("Please enter a message.")
            else:
                pred, spam_prob           = classify(user_input, model, vectorizer)
                cats                      = detect_categories(user_input)
                hl                        = highlight_html(user_input, cats)
                st.session_state.msg      = user_input
                st.session_state.pred     = pred
                st.session_state.spam_prob= spam_prob
                st.session_state.categories = cats
                st.session_state.highlighted= hl
                st.session_state.view     = "results"
                st.rerun()

    # ── RESULTS VIEW ────────────────────────────────────────
    else:
        text        = st.session_state.msg
        spam_prob   = st.session_state.spam_prob
        cats        = st.session_state.categories
        hl          = st.session_state.highlighted
        label, lcolor = score_label(spam_prob)

        left, right = st.columns([6, 4], gap="large")

        # ── LEFT: highlighted text document ────────────────
        with left:
            st.markdown(
                f"""
                <div style="
                    background:#FAFAF8;
                    border:1.5px solid #D4D9BE;
                    border-radius:10px;
                    padding:1.8rem 2rem;
                    font-size:1rem;
                    line-height:1.85;
                    color:#1C1C1A;
                    min-height:260px;
                    white-space:pre-wrap;
                    word-break:break-word;
                ">{hl}</div>
                """,
                unsafe_allow_html=True
            )

            if st.button("← Analyse another message"):
                st.session_state.view = "input"
                st.rerun()

        # ── RIGHT: score panel ──────────────────────────────
        with right:

            # Overall score card
            st.markdown(
                f"""
                <div style="
                    border:1.5px solid #E4E8D4;
                    border-radius:10px;
                    padding:1.6rem 1.4rem 1.2rem;
                    background:#FAFAF8;
                ">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                    <span style="font-size:0.9rem;color:#6B6B65;font-weight:500;">Overall score</span>
                    <span style="font-size:1rem;font-weight:700;color:{lcolor};">{label}</span>
                  </div>

                  <!-- Score bar -->
                  <div style="background:#E4E8D4;border-radius:4px;height:8px;width:100%;margin-bottom:1.2rem;">
                    <div style="background:{lcolor};border-radius:4px;height:8px;width:{spam_prob*100:.0f}%;"></div>
                  </div>

                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;">
                    <div>
                      <div style="font-size:0.78rem;color:#6B6B65;">Spam probability</div>
                      <div style="font-size:1.3rem;font-weight:700;font-family:'DM Mono',monospace;color:{lcolor};">{spam_prob*100:.1f}%</div>
                    </div>
                    <div>
                      <div style="font-size:0.78rem;color:#6B6B65;">Words</div>
                      <div style="font-size:1.3rem;font-weight:700;font-family:'DM Mono',monospace;color:#1C1C1A;">{word_count(text)}</div>
                    </div>
                    <div style="grid-column:1/-1;">
                      <div style="font-size:0.78rem;color:#6B6B65;">Read time</div>
                      <div style="font-size:1rem;font-weight:600;color:#1C1C1A;">{read_time(text)}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Category breakdown
            if cats:
                st.write("")
                st.markdown(
                    "<p style='font-size:0.82rem;color:#6B6B65;font-weight:600;"
                    "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;'>"
                    "Signal breakdown</p>",
                    unsafe_allow_html=True
                )
                for cat, matches in cats.items():
                    cfg = CATEGORIES[cat]
                    st.markdown(
                        f"""
                        <div style="
                            display:flex;align-items:center;justify-content:space-between;
                            padding:0.65rem 1rem;
                            border:1.5px solid {cfg['border']}44;
                            background:{cfg['bg']};
                            border-radius:8px;
                            margin-bottom:0.5rem;
                        ">
                          <span style="font-size:0.95rem;font-weight:500;color:{cfg['text']};">
                            {cfg['emoji']} &nbsp;{cat}
                          </span>
                          <span style="
                            background:{'#FFFFFF'};
                            border:1.5px solid {cfg['border']};
                            color:{cfg['text']};
                            font-family:'DM Mono',monospace;
                            font-weight:600;
                            font-size:0.85rem;
                            padding:1px 8px;
                            border-radius:20px;
                          ">({len(matches)})</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.write("")
                st.markdown(
                    "<div style='border:1.5px solid #D4D9BE;border-radius:8px;"
                    "padding:1rem;text-align:center;color:#6B6B65;font-size:0.9rem;'>"
                    "✅ No suspicious signal categories detected.</div>",
                    unsafe_allow_html=True
                )

        # ── LEGEND: display below both columns ──────────────
        if cats:
            st.write("")
            legend_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;'>"
            for cat in cats:
                cfg = CATEGORIES[cat]
                legend_html += (
                    f"<span style='background:{cfg['bg']};border:1.5px solid {cfg['border']};"
                    f"color:{cfg['text']};padding:4px 12px;border-radius:20px;"
                    f"font-size:0.85rem;font-weight:500;'>"
                    f"{cfg['emoji']} {cat}</span>"
                )
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

# ── ═════════════════════════════════════════════════════════
# PERFORMANCE TAB
# ── ═════════════════════════════════════════════════════════

OLIVE      = "#4A5A2A"
OLIVE_MID  = "#6B7A3E"
OLIVE_PALE = "#D4D9BE"
RUST       = "#B85C3A"
BG         = "#FFFFFF"
TEXT       = "#1C1C1A"
MUTED      = "#6B6B65"

def chart_style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(OLIVE_PALE)
    ax.spines['bottom'].set_color(OLIVE_PALE)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

with perf_tab:
    st.markdown("#### Model metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{metrics['accuracy']  * 100:.2f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    m3.metric("Recall",    f"{metrics['recall']    * 100:.2f}%")
    m4.metric("F1 Score",  f"{metrics['f1']        * 100:.2f}%")

    st.caption(
        "**Precision** — of all flagged spam, how many were actually spam?  "
        "**Recall** — of all real spam, how many did the model catch?  "
        "**F1** — harmonic mean of the two; the key metric on imbalanced data."
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Confusion matrix")
        cm = metrics["cm"]
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        from matplotlib.patches import Rectangle
        colors = [[OLIVE_PALE, RUST + "44"], [RUST + "44", OLIVE + "CC"]]
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color=colors[i][j], zorder=1))
        labels = [["TN","FP"],["FN","TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                        ha="center", va="center", fontsize=14,
                        fontweight="600", color=TEXT, zorder=2)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Ham","Spam"], fontsize=11, color=TEXT)
        ax.set_yticklabels(["Ham","Spam"], fontsize=11, color=TEXT)
        ax.set_xlabel("Predicted", color=MUTED, fontsize=10)
        ax.set_ylabel("Actual",    color=MUTED, fontsize=10)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        for sp in ax.spines.values(): sp.set_color(OLIVE_PALE)
        chart_style(ax, fig)
        st.pyplot(fig); plt.close()
        st.caption("**TN** correct ham · **TP** correct spam · **FP** false alarm · **FN** missed spam")

    with c2:
        st.markdown("#### Top spam-indicating words")
        words, scores = zip(*top_spam[:12])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        pos = range(len(words))
        ax.barh(list(reversed(list(pos))), list(reversed(list(scores))),
                color=OLIVE_MID, height=0.6)
        ax.set_yticks(list(pos))
        ax.set_yticklabels(list(reversed(words)), fontsize=9, color=TEXT)
        ax.set_xlabel("Model coefficient", color=MUTED, fontsize=9)
        chart_style(ax, fig)
        st.pyplot(fig); plt.close()
        st.caption("Higher coefficient = stronger spam signal in the logistic regression model.")

    st.divider()
    st.markdown("#### Dataset")
    spam_n = int(df["label"].sum())
    ham_n  = len(df) - spam_n
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total",      f"{len(df):,}")
    d2.metric("Ham",        f"{ham_n:,}")
    d3.metric("Spam",       f"{spam_n:,}")
    d4.metric("Spam ratio", f"{spam_n/len(df)*100:.1f}%")

    # Native, scale-proof dataset percentage HTML replacing matplotlib:
    ham_pct = (ham_n / len(df)) * 100
    spam_pct = (spam_n / len(df)) * 100
    st.markdown(
        f"""
        <div style="width: 100%; height: 28px; display: flex; border-radius: 8px; overflow: hidden; margin-top: 18px; margin-bottom: 12px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">
            <div style="width: {ham_pct}%; background-color: {OLIVE};"></div>
            <div style="width: {spam_pct}%; background-color: {RUST};"></div>
        </div>

        <div style="display: flex; justify-content: center; gap: 24px; margin-top: 10px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 14px; height: 14px; background-color: {OLIVE}; border-radius: 3px;"></div>
                <span style="font-size: 0.95rem; color: {TEXT}; font-weight: 500;">Ham</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 14px; height: 14px; background-color: {RUST}; border-radius: 3px;"></div>
                <span style="font-size: 0.95rem; color: {TEXT}; font-weight: 500;">Spam</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════
# ABOUT TAB
# ═══════════════════════════════════════════════════════════

with about_tab:
    st.markdown("#### How this works")
    st.markdown("""
    **Dataset:** SMS Spam Collection — 5,574 messages (4,827 ham · 747 spam) from UCI ML Repository.

    **Preprocessing pipeline:**
    1. Strip special characters via regex
    2. Lowercase
    3. Remove NLTK English stopwords
    4. Porter stemming — "winning" → "win", "guaranteed!" → "guaranty"

    **Features:** TF-IDF with 5,000 features and bigrams `(1,2)-ngrams`.
    Rewards words common in *this* message but rare across *all* messages.

    **Classifier:** Logistic Regression (C=5, L2 regularization).
    Outputs a calibrated probability score, not just a binary label.

    **Category detection** is a separate rule-based layer on top of the ML model —
    regex patterns for Urgency, Shady, Financial, and Overpromise language.
    These explain *why* a message looks like spam, independent of the model score.

    **Why precision over recall:** A false positive (legitimate email flagged as spam)
    is more disruptive than a false negative. The model is tuned accordingly.

    ---
    **Stack:** Scikit-learn · NLTK · Streamlit · Matplotlib · Pandas

    Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) ·
    [GitHub](https://github.com/garvranaaa) ·
    [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
    """)