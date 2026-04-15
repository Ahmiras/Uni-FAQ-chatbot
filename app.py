import streamlit as st
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(
    page_title="UniBot - University FAQ",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
    }

    .main-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }

    .header-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .header-sub {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .chat-container {
        background: #f8f9ff;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        min-height: 380px;
        max-height: 420px;
        overflow-y: auto;
        border: 1px solid #e8e8ff;
    }

    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.75rem 1.1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem 3rem;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
        word-wrap: break-word;
    }

    .bot-bubble {
        background: white;
        color: #2d2d2d;
        padding: 0.75rem 1.1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 3rem 0.5rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
        border: 1px solid #e8e8ff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        word-wrap: break-word;
    }

    .bubble-label-user {
        text-align: right;
        font-size: 0.72rem;
        color: #aaa;
        margin-bottom: 2px;
        padding-right: 4px;
    }

    .bubble-label-bot {
        text-align: left;
        font-size: 0.72rem;
        color: #aaa;
        margin-bottom: 2px;
        padding-left: 4px;
    }

    .quick-btn {
        display: inline-block;
        background: linear-gradient(135deg, #f8f9ff, #e8e8ff);
        color: #667eea;
        border: 1.5px solid #c5c9ff;
        border-radius: 20px;
        padding: 0.35rem 0.85rem;
        margin: 0.25rem;
        font-size: 0.82rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 500;
    }

    .quick-btn:hover {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-color: transparent;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }

    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.8rem 0 0.4rem;
    }

    .confidence-bar {
        height: 5px;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin-top: 4px;
    }

    .stat-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 1px solid #c5c9ff;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        text-align: center;
    }

    .stat-num {
        font-family: 'Sora', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #999;
        margin-top: 2px;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 14px !important;
        border: 2px solid #c5c9ff !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s ease !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100%;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(102,126,234,0.4) !important;
    }

    .stAlert {
        border-radius: 12px !important;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #c5c9ff; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'not', 'no', 'when', 'where', 'what', 'how', 'who', 'which'}

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    responses = pickle.load(open("responses.pkl", "rb"))
    return model, responses

try:
    model, responses = load_model()
    model_loaded = True
except:
    model_loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "bot",
        "content": "👋 Hello! I'm UniBot, your university assistant. Ask me about admissions, fees, courses, scholarships, and more!"
    })

if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

if "last_intent" not in st.session_state:
    st.session_state.last_intent = "-"

def get_response(user_input):
    processed = preprocess(user_input)
    proba = model.predict_proba([processed])[0]
    max_proba = max(proba)
    intent = model.predict([processed])[0]

    if max_proba < 0.25:
        return "I'm not sure I understood that. Could you rephrase? You can ask about admissions, fees, courses, scholarships, hostel, timing, or contact info.", max_proba, "unknown"

    reply = random.choice(responses[intent])
    return reply, max_proba, intent

quick_questions = [
    "How to apply?",
    "What is the fee?",
    "Scholarship info",
    "Hostel available?",
    "Contact details",
    "Library timings",
    "Transport service",
    "Student clubs"
]

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="header-title">🎓 UniBot</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Your 24/7 University FAQ Assistant</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'''<div class="stat-box">
        <div class="stat-num">{st.session_state.msg_count}</div>
        <div class="stat-label">Messages</div>
    </div>''', unsafe_allow_html=True)
with col2:
    st.markdown(f'''<div class="stat-box">
        <div class="stat-num">{int(st.session_state.last_confidence * 100)}%</div>
        <div class="stat-label">Confidence</div>
    </div>''', unsafe_allow_html=True)
with col3:
    st.markdown(f'''<div class="stat-box">
        <div class="stat-num">16</div>
        <div class="stat-label">Topics Known</div>
    </div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

chat_html = '<div class="chat-container">'
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f'<div class="bubble-label-user">You</div><div class="user-bubble">{msg["content"]}</div>'
    else:
        chat_html += f'<div class="bubble-label-bot">🤖 UniBot</div><div class="bot-bubble">{msg["content"]}</div>'
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

st.markdown('<div class="section-label">Quick Questions</div>', unsafe_allow_html=True)
cols = st.columns(4)
for i, q in enumerate(quick_questions):
    with cols[i % 4]:
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            if model_loaded:
                st.session_state.messages.append({"role": "user", "content": q})
                reply, conf, intent = get_response(q)
                st.session_state.messages.append({"role": "bot", "content": reply})
                st.session_state.msg_count += 1
                st.session_state.last_confidence = conf
                st.session_state.last_intent = intent
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
col_input, col_btn = st.columns([4, 1])
with col_input:
    user_input = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed", key="user_input")
with col_btn:
    send = st.button("Send ➤")

if send and user_input.strip():
    if model_loaded:
        st.session_state.messages.append({"role": "user", "content": user_input})
        reply, conf, intent = get_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": reply})
        st.session_state.msg_count += 1
        st.session_state.last_confidence = conf
        st.session_state.last_intent = intent
        st.rerun()
    else:
        st.error("⚠️ Model not found! Please run `python train.py` first.")

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = [{"role": "bot", "content": "👋 Hello! I'm UniBot, your university assistant. Ask me about admissions, fees, courses, scholarships, and more!"}]
    st.session_state.msg_count = 0
    st.session_state.last_confidence = 0.0
    st.session_state.last_intent = "-"
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color: rgba(255,255,255,0.7); font-size: 0.8rem; margin-top: 1rem;">
    Built with ❤️ using Scikit-learn & Streamlit &nbsp;|&nbsp; ML Pre-Mid Project
</div>
""", unsafe_allow_html=True)