import streamlit as st
import numpy as np
import joblib
import requests

# -----------------------------
# Backend Functions
# -----------------------------

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": text_list}
    )
    return r.json()["embeddings"]


def ask_llm(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi", "prompt": prompt, "stream": False}
    )
    return r.json()["response"]


# -----------------------------
# Load Embeddings
# -----------------------------

df = joblib.load("embeddings.joblib")

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="NEXUS · RAG Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# DARK ANIMATED CSS
# -----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Orbitron:wght@400;700;900&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-void:       #020408;
    --bg-deep:       #060d14;
    --bg-surface:    #0a1520;
    --bg-card:       #0d1d2e;
    --bg-elevated:   #112338;
    --accent-cyan:   #00d4ff;
    --accent-blue:   #0066ff;
    --accent-purple: #7c3aed;
    --accent-green:  #00ff88;
    --accent-amber:  #ffb700;
    --text-primary:  #e8f4ff;
    --text-secondary:#8aafc8;
    --text-muted:    #3d6080;
    --border-dim:    rgba(0, 212, 255, 0.08);
    --border-glow:   rgba(0, 212, 255, 0.25);
    --glow-cyan:     0 0 20px rgba(0, 212, 255, 0.3), 0 0 60px rgba(0, 212, 255, 0.1);
    --glow-blue:     0 0 20px rgba(0, 102, 255, 0.3), 0 0 60px rgba(0, 102, 255, 0.1);
    --glow-green:    0 0 20px rgba(0, 255, 136, 0.25);
}

/* ===== GLOBAL RESET ===== */
html, body, .stApp {
    background: var(--bg-void) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ===== ANIMATED GRID BACKGROUND ===== */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridShift 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes gridShift {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}

/* ===== ANIMATED PARTICLES (corner orbs) ===== */
.stApp::after {
    content: '';
    position: fixed;
    width: 600px;
    height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,102,255,0.06) 0%, transparent 70%);
    top: -200px;
    right: -200px;
    animation: orbPulse 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes orbPulse {
    0%   { transform: scale(1) translate(0,0); opacity: 0.5; }
    100% { transform: scale(1.3) translate(-30px, 30px); opacity: 1; }
}

/* ===== MAIN CONTENT ABOVE OVERLAYS ===== */
.main .block-container {
    position: relative;
    z-index: 1;
    padding: 0 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ===== HEADER SECTION ===== */
.nexus-header {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}

.nexus-logo {
    font-family: 'Orbitron', monospace;
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-blue) 50%, var(--accent-purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s ease-in-out infinite;
    margin-bottom: 0.5rem;
}

@keyframes shimmer {
    0%, 100% { filter: brightness(1); }
    50%       { filter: brightness(1.3); }
}

.nexus-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.3em;
    color: var(--accent-cyan);
    opacity: 0.6;
    text-transform: uppercase;
}

.nexus-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-blue), transparent);
    margin: 1.5rem auto;
    max-width: 600px;
    position: relative;
    animation: scanLine 3s ease-in-out infinite;
}

@keyframes scanLine {
    0%,100% { opacity: 0.4; }
    50%      { opacity: 1; }
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: var(--bg-deep) !important;
    border-right: 1px solid var(--border-dim) !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(180deg, rgba(0,212,255,0.03) 0%, transparent 40%);
    pointer-events: none;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.2em !important;
    color: var(--accent-cyan) !important;
    text-transform: uppercase;
}

/* ===== SLIDER STYLING ===== */
[data-testid="stSlider"] {
    padding: 0.5rem 0 !important;
}

[data-testid="stSlider"] .stSlider > div > div > div {
    background: var(--accent-cyan) !important;
}

/* ===== SIDEBAR PANEL BOX ===== */
.sidebar-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.sidebar-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
}

/* ===== CHAT MESSAGES ===== */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.75rem 0 !important;
    animation: fadeSlideIn 0.4s ease-out forwards;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* USER bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-left: 3rem !important;
    box-shadow: inset 0 1px 0 rgba(0,212,255,0.1);
}

/* ASSISTANT bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-right: 3rem !important;
    box-shadow: var(--glow-cyan);
}

/* ===== CHAT AVATARS ===== */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    border-radius: 8px !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)) !important;
    border-radius: 8px !important;
}

/* ===== CHAT INPUT ===== */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 16px !important;
    box-shadow: var(--glow-cyan) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ===== CONFIDENCE BADGE ===== */
.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent-cyan);
    margin-bottom: 1rem;
    animation: badgePulse 2s ease-in-out infinite;
}

@keyframes badgePulse {
    0%,100% { box-shadow: 0 0 8px rgba(0,212,255,0.15); }
    50%      { box-shadow: 0 0 20px rgba(0,212,255,0.35); }
}

.confidence-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: dotBlink 1.5s ease-in-out infinite;
}

.confidence-dot.low {
    background: #ff4444;
}

@keyframes dotBlink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.2; }
}

/* ===== EXPANDER (Chunk cards) ===== */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stExpander"]:hover {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.1) !important;
}

[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--accent-cyan) !important;
    padding: 0.75rem 1rem !important;
    background: var(--bg-surface) !important;
    border-bottom: 1px solid var(--border-dim) !important;
}

[data-testid="stExpander"] summary:hover {
    background: var(--bg-elevated) !important;
}

/* ===== SUBHEADERS ===== */
.stMarkdown h2, .stMarkdown h3 {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.25em !important;
    color: var(--accent-cyan) !important;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem !important;
}

/* ===== ERROR / WARNING ===== */
[data-testid="stAlert"] {
    background: rgba(255, 68, 68, 0.08) !important;
    border: 1px solid rgba(255, 68, 68, 0.3) !important;
    border-radius: 12px !important;
    color: #ff8888 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ===== RESPONSE BLOCK ===== */
.response-block {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    line-height: 1.75;
    color: var(--text-primary);
    position: relative;
    overflow: hidden;
}

.response-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 100%;
    background: linear-gradient(180deg, rgba(0,212,255,0.02) 0%, transparent 40%);
    pointer-events: none;
}

/* ===== SECTION LABELS ===== */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: var(--accent-cyan);
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-glow), transparent);
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* ===== WRITE ANIMATION ===== */
.typing-indicator {
    display: flex;
    gap: 4px;
    align-items: center;
    padding: 0.5rem;
}
.typing-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-cyan);
    animation: typingBounce 1.2s ease-in-out infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce {
    0%,80%,100% { transform: scale(0.6); opacity: 0.4; }
    40%          { transform: scale(1.0); opacity: 1; }
}

/* ===== PULSE RING ANIMATION ===== */
@keyframes pulseRing {
    0%   { transform: scale(0.8); opacity: 0.8; }
    100% { transform: scale(2); opacity: 0; }
}

/* ===== STATUS INDICATOR ===== */
.status-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    padding: 0.5rem 0;
    margin-bottom: 1rem;
}

.status-dot-active {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 6px var(--accent-green);
    animation: dotBlink 2s ease-in-out infinite;
    flex-shrink: 0;
}

/* ===== CHUNK METADATA TAGS ===== */
.time-tag {
    display: inline-block;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-cyan);
    margin-right: 4px;
}

/* ===== OVERRIDE STREAMLIT DEFAULTS ===== */
.stMarkdown p { color: var(--text-primary) !important; line-height: 1.7 !important; }
.stMarkdown code {
    background: var(--bg-elevated) !important;
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
    padding: 0.1em 0.4em !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ===== SIDEBAR METRIC CARD ===== */
.metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: var(--accent-cyan);
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------

st.markdown("""
<div class="nexus-header">
    <div class="nexus-logo">NEXUS</div>
    <div class="nexus-subtitle">⬡ &nbsp; Retrieval · Augmented · Intelligence &nbsp; ⬡</div>
    <div class="nexus-divider"></div>
    <div class="status-bar" style="justify-content:center">
        <div class="status-dot-active"></div>
        <span>VECTOR DATABASE ONLINE</span>
        <span style="opacity:0.3">│</span>
        <span>MODEL: PHI · EMBEDDER: BGE-M3</span>
        <span style="opacity:0.3">│</span>
        <span>OLLAMA: CONNECTED</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem">
        <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem; letter-spacing: 0.3em;
                    color: #00d4ff; text-transform: uppercase; margin-bottom: 0.5rem">
            ⬡ Control Panel
        </div>
        <div style="height:1px; background: linear-gradient(90deg, #00d4ff44, transparent); margin-bottom:1.5rem"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Retrieval Config</p>', unsafe_allow_html=True)

    top_k = st.slider("Chunks to Retrieve", 1, 5, 3,
                      help="Number of video segments to surface per query")

    threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.35,
                          help="Minimum cosine similarity to accept a result")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Session Stats</p>', unsafe_allow_html=True)

    msg_count = len(st.session_state.get("messages", []))
    q_count = msg_count // 2

    st.markdown(f"""
    <div class="metric-card">
        <span class="metric-label">Queries Sent</span>
        <span class="metric-value">{q_count:02d}</span>
    </div>
    <div class="metric-card">
        <span class="metric-label">Top-K</span>
        <span class="metric-value">{top_k}</span>
    </div>
    <div class="metric-card">
        <span class="metric-label">Min Score</span>
        <span class="metric-value">{threshold:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">About</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color: #3d6080; font-family: 'JetBrains Mono', monospace;
                line-height: 1.8; padding: 0.5rem 0">
        Semantic search over<br>
        video subtitle chunks<br>
        powered by local LLM<br>
        + vector similarity.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Chat Memory
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input("⬡  Query the video knowledge base...")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # ---- RAG Retrieval ----

    question_embedding = create_embedding([question])[0]
    embeddings_matrix = np.vstack(df['embedding'])

    similarities = (
        np.dot(embeddings_matrix, question_embedding)
        / (
            np.linalg.norm(embeddings_matrix, axis=1)
            * np.linalg.norm(question_embedding)
        )
    )

    max_indx = similarities.argsort()[::-1][:top_k]
    new_df = df.loc[max_indx]
    best_score = similarities[max_indx[0]]

    # ---- AI Response ----

    with st.chat_message("assistant"):

        dot_class = "confidence-dot" if best_score >= threshold else "confidence-dot low"
        score_color = "#00ff88" if best_score >= threshold else "#ff4444"

        st.markdown(f"""
        <div class="confidence-badge">
            <div class="{dot_class}"></div>
            COSINE SIMILARITY &nbsp;·&nbsp;
            <span style="color:{score_color}; font-weight:600">{best_score:.4f}</span>
            &nbsp;·&nbsp;
            <span style="opacity:0.5">{"MATCH" if best_score >= threshold else "BELOW THRESHOLD"}</span>
        </div>
        """, unsafe_allow_html=True)

        if best_score < threshold:
            response = "⬡ Query vector falls outside dataset manifold — no relevant segments found."
            st.error(response)

        else:

            st.markdown('<p class="section-label">Relevant Segments</p>', unsafe_allow_html=True)

            for i, (_, row) in enumerate(new_df.iterrows()):
                score_i = similarities[max_indx[i]]
                with st.expander(f"⟨ {row['start']}s → {row['end']}s ⟩  · score {score_i:.3f}"):
                    st.markdown(f"""
                    <span class="time-tag">START {row['start']}s</span>
                    <span class="time-tag">END {row['end']}s</span>
                    <br><br>
                    <span style="font-size:0.9rem; color:#c8dff0; line-height:1.8">{row['text']}</span>
                    """, unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="margin-top:1.5rem">AI Response</p>',
                        unsafe_allow_html=True)

            prompt = f"""
            You are an AI assistant helping with video content.

            Relevant subtitle chunks:
            {new_df[['start','end','text']].to_json(orient='records')}

            User Question:
            {question}

            Answer clearly using the chunks.
            Mention timestamps if useful.
            """

            response = ask_llm(prompt)

            st.markdown(f"""
            <div class="response-block">
                {response}
            </div>
            """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
