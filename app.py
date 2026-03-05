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

st.set_page_config(page_title="RAG Video Assistant", page_icon="🎥", layout="wide")

st.title("🎥 RAG Video Assistant")

# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.title("⚙ Settings")

top_k = st.sidebar.slider("No. of Chunks", 1, 5, 3)
threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.35)

# -----------------------------
# Chat Memory
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input("Ask something about your videos...")

if question:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    # -----------------------------
    # RAG Retrieval
    # -----------------------------

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

    # -----------------------------
    # AI Response Block
    # -----------------------------

    with st.chat_message("assistant"):

        st.write(f"**Confidence Score:** {best_score:.2f}")

        if best_score < threshold:
            response = "Question not related to dataset 😌"
            st.error(response)

        else:

            st.subheader("📌 Relevant Chunks")

            for _, row in new_df.iterrows():
                with st.expander(f"⏱ {row['start']}s → {row['end']}s"):
                    st.write(row['text'])

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

            st.subheader("🤖 AI Response")
            st.write(response)

    # Save AI message
    st.session_state.messages.append({"role": "assistant", "content": response})