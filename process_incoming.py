import json
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
import os

CHAT_HISTORY_FILE = "chat_history.json"

def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    embedding = r.json()["embeddings"]
    return embedding


def ask_llm(prompt):

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": prompt,
            "stream": False
        }
    )

    return r.json()["response"]


# Load embeddings
df = joblib.load('embeddings.joblib')

if os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "r") as f:
        chat_history = json.load(f)
else:
    chat_history = []

# Ask Question
incoming_query = input("Ask a Question: ")

question_embedding = create_embedding([incoming_query])[0]


# Compute Similarity
similarities = cosine_similarity(
    np.vstack(df['embedding'].values),
    [question_embedding]
).flatten()


# Top Matches
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]


# Confidence Score
best_score = similarities[max_indx[0]]
print(f"\nConfidence Score: {best_score:.2f}")

if best_score > 0.75:
    print("Confidence Level: HIGH ✅")

elif best_score > 0.50:
    print("Confidence Level: MEDIUM 👍")

else:
    print("Confidence Level: LOW ")


# Show Retrieved Chunks
print("\nTop Matching Chunks:\n")

for i, (_, row) in enumerate(new_df.iterrows()):

    score = similarities[max_indx[i]]

    if i == 0:
        print("\n⭐ BEST MATCH")

    print(f"\nChunk ID: {row['chunk_id']}")
    print(f"Similarity Score: {score:.2f}")
    print(f"Source: {row['source']}")

    print(f"[{row['start']}s → {row['end']}s]")
    print(row['text'])

# Build Prompt
prompt = f"""
You are an AI assistant helping with video content.

Previous conversation:
{json.dumps(chat_history[-3:], indent=2)}

Relevant subtitle chunks:
{new_df[["start", "end", "text"]].to_json(orient="records")}

User Question:
{incoming_query}

Answer clearly using the chunks.
"""
# Ask LLM
response = ask_llm(prompt)
chat_history.append({
    "question": incoming_query,
    "response": response
})

with open(CHAT_HISTORY_FILE, "w") as f:
    json.dump(chat_history, f, indent=4)

print("\nAI Response:\n")
print(response)