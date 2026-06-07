import streamlit as st
import joblib
import re
import nltk

# Download NLTK resources only if missing
resources = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4"
}

for path, resource in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load Saved Model
saved_model = joblib.load("text_model.pkl")

model = saved_model['model']
TFIDF = saved_model['TFIDF']
LE = saved_model['Label_encod']

# NLP Setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text Cleaning Function — UNCHANGED
def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --ink:       #0d0d0d;
    --paper:     #f7f4ef;
    --cream:     #ede9e1;
    --accent:    #c8102e;
    --accent2:   #1a3a5c;
    --muted:     #6b6560;
    --border:    #d6d0c8;
    --card-bg:   #ffffff;
    --shadow:    0 2px 20px rgba(0,0,0,0.07);
    --shadow-lg: 0 8px 48px rgba(0,0,0,0.12);
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--paper) !important;
    color: var(--ink) !important;
}

.stApp {
    background-color: var(--paper) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 780px !important;
}

/* ── Masthead ── */
.masthead {
    text-align: center;
    border-top: 4px solid var(--accent);
    border-bottom: 2px solid var(--ink);
    padding: 2.5rem 1rem 2rem;
    margin-bottom: 0.5rem;
    background: var(--paper);
    position: relative;
}
.masthead-overline {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    font-weight: 900;
    line-height: 1.05;
    color: var(--ink);
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.masthead-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.0rem;
    font-weight: 300;
    color: var(--muted);
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}
.masthead-rule {
    width: 40px;
    height: 3px;
    background: var(--accent);
    margin: 1.2rem auto 0;
    border-radius: 2px;
}

/* ── Date / Edition line ── */
.edition-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding: 0.55rem 0;
    margin-bottom: 2rem;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Instruction card ── */
.instruction-card {
    background: var(--cream);
    border-left: 3px solid var(--accent2);
    border-radius: 4px;
    padding: 0.85rem 1.1rem;
    font-size: 0.87rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
    line-height: 1.65;
}
.instruction-card strong {
    color: var(--ink);
    font-weight: 600;
}

/* ── Text area ── */
.stTextArea textarea {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--ink) !important;
    padding: 1rem !important;
    line-height: 1.7 !important;
    transition: border-color 0.2s ease !important;
    box-shadow: var(--shadow) !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent2) !important;
    box-shadow: 0 0 0 3px rgba(26,58,92,0.08) !important;
    outline: none !important;
}
.stTextArea label { display: none !important; }

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 1.2rem;
    margin: 0.6rem 0 1.5rem;
}
.stat-chip {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.28rem 0.85rem;
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 500;
}
.stat-chip span {
    color: var(--ink);
    font-weight: 600;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: var(--ink) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: background 0.18s ease, transform 0.12s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}
.stButton > button:hover {
    background: var(--accent2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,58,92,0.25) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 8px;
    padding: 2rem 2rem 1.8rem;
    margin-top: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    animation: slideUp 0.4s cubic-bezier(0.22, 1, 0.36, 1);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
}

/* Category-specific themes */
.cat-business   { background: #fff8f0; border: 1.5px solid #f0dcc0; }
.cat-business::before { background: linear-gradient(90deg, #d4821a, #e8a94d); }

.cat-entertainment { background: #fdf0f7; border: 1.5px solid #e8c0d8; }
.cat-entertainment::before { background: linear-gradient(90deg, #9b2d7a, #c45fa0); }

.cat-politics   { background: #f0f4ff; border: 1.5px solid #bfcfee; }
.cat-politics::before { background: linear-gradient(90deg, #1a3a5c, #2e63a0); }

.cat-sport      { background: #f0faf2; border: 1.5px solid #b8dfc2; }
.cat-sport::before { background: linear-gradient(90deg, #1a7a3c, #2daa5c); }

.cat-tech       { background: #f0f8ff; border: 1.5px solid #b8d8f0; }
.cat-tech::before { background: linear-gradient(90deg, #0a5fa0, #1a8fd4); }

.result-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.7rem;
}
.result-category {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.result-message {
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.6;
    border-top: 1px solid var(--border);
    padding-top: 0.9rem;
    margin-top: 0.9rem;
}

/* ── Warning / empty state ── */
.warn-box {
    background: #fff9ec;
    border: 1.5px solid #f0d080;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    font-size: 0.9rem;
    color: #7a5a00;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1rem;
    animation: slideUp 0.3s ease;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.8;
}
.app-footer a {
    color: var(--accent2);
    text-decoration: none;
    font-weight: 500;
}

/* ── Spinner tweak ── */
.stSpinner > div { border-top-color: var(--accent2) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-overline">📰 Machine Learning · NLP</div>
    <h1 class="masthead-title">BBC News<br>Classifier</h1>
    <p class="masthead-subtitle">
        Paste any news article and the model will instantly identify
        which of the five BBC editorial desks it belongs to.
    </p>
    <div class="masthead-rule"></div>
</div>
""", unsafe_allow_html=True)

# Edition bar
import datetime
today = datetime.date.today().strftime("%A, %d %B %Y")
st.markdown(f"""
<div class="edition-bar">
    <span>{today}</span>
    <span>NLP Classifier v1.0</span>
</div>
""", unsafe_allow_html=True)


# ─── Input Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Article Input</div>', unsafe_allow_html=True)

st.markdown("""
<div class="instruction-card">
    <strong>How to use:</strong> Paste or type a news article in the box below — 
    a headline, a paragraph, or the full piece. Click <strong>Classify Article</strong> 
    and the model will determine the most likely category.
</div>
""", unsafe_allow_html=True)

PLACEHOLDER = (
    "e.g. The Prime Minister announced a new economic stimulus package today, "
    "pledging £5 billion toward green infrastructure projects across the UK. "
    "Opposition leaders questioned the funding source and called for an emergency debate…"
)

user_text = st.text_area(
    label="news_input",
    placeholder=PLACEHOLDER,
    height=220,
    key="article_input"
)

# Live stats
char_count = len(user_text)
word_count = len(user_text.split()) if user_text.strip() else 0

st.markdown(f"""
<div class="stats-row">
    <div class="stat-chip">Characters: <span>{char_count:,}</span></div>
    <div class="stat-chip">Words: <span>{word_count:,}</span></div>
    <div class="stat-chip">Min. recommended: <span>30 words</span></div>
</div>
""", unsafe_allow_html=True)


# ─── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Classify Article")


# ─── Category Config ───────────────────────────────────────────────────────────
CATEGORY_CONFIG = {
    "business": {
        "emoji": "💼",
        "css_class": "cat-business",
        "color": "#d4821a",
        "message": "This article covers topics such as markets, economics, corporate news, or financial policy."
    },
    "entertainment": {
        "emoji": "🎬",
        "css_class": "cat-entertainment",
        "color": "#9b2d7a",
        "message": "This article covers topics such as film, music, celebrity news, arts, or pop culture."
    },
    "politics": {
        "emoji": "🏛️",
        "css_class": "cat-politics",
        "color": "#1a3a5c",
        "message": "This article covers topics such as government, elections, policy, legislation, or international affairs."
    },
    "sport": {
        "emoji": "⚽",
        "css_class": "cat-sport",
        "color": "#1a7a3c",
        "message": "This article covers topics such as football, athletics, competitions, transfers, or sporting results."
    },
    "tech": {
        "emoji": "💻",
        "css_class": "cat-tech",
        "color": "#0a5fa0",
        "message": "This article covers topics such as software, gadgets, AI, cybersecurity, or the tech industry."
    },
}


# ─── Prediction Logic ──────────────────────────────────────────────────────────
if predict_clicked:
    if not user_text.strip():
        st.markdown("""
        <div class="warn-box">
            ⚠️ Please paste or type a news article before classifying.
        </div>
        """, unsafe_allow_html=True)

    elif word_count < 5:
        st.markdown("""
        <div class="warn-box">
            ⚠️ The article seems too short. Try adding more text for a reliable prediction.
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Analysing article…"):
            # ── BACKEND UNCHANGED ──────────────────────────────────────
            cleaned_text      = text_cleaning(user_text)
            transformed_text  = TFIDF.transform([cleaned_text])
            prediction        = model.predict(transformed_text)
            category          = LE.inverse_transform(prediction)[0].lower()
            # ── END BACKEND ────────────────────────────────────────────

        cfg = CATEGORY_CONFIG.get(category, {
            "emoji": "📄",
            "css_class": "",
            "color": "#333",
            "message": "Category identified."
        })

        st.markdown(f"""
        <div class="result-card {cfg['css_class']}">
            <div class="result-eyebrow">✅ Classification complete</div>
            <div class="result-category" style="color:{cfg['color']}">
                {cfg['emoji']} {category.capitalize()}
            </div>
            <div class="result-message">
                {cfg['message']}
                <br><br>
                Article analysed — <strong>{word_count:,} words</strong>, 
                <strong>{char_count:,} characters</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built with <strong>Python</strong> · <strong>Scikit-learn</strong> · <strong>NLTK</strong> · <strong>Streamlit</strong>
    <br>
    KNN classifier trained on the BBC News dataset &nbsp;·&nbsp; TF-IDF text vectorisation
</div>
""", unsafe_allow_html=True)