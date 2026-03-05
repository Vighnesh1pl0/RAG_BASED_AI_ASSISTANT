import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib


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


# Read JSON chunk files
jsons = os.listdir("json")

my_dicts = []
chunk_id = 0


for json_file in jsons:

    with open(f"json/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"Processing {json_file}")

    # Create embeddings for chunk texts
    embeddings = create_embedding([c['text'] for c in content])

    for i, chunk in enumerate(content):

        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]

        chunk_id += 1
        my_dicts.append(chunk)


# Convert → DataFrame
df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, 'embeddings.joblib')
