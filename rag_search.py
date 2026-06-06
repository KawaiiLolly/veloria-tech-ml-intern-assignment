"""
    rag_search.py
"""

import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer

df = pd.read_csv("match_data.csv")

sentences = []

for _, row in df.iterrows():

    sentence = (
        f"{row['team1']} vs {row['team2']} "
        f"at {row['venue']} on {row['match_date']}. "
        f"{row['winner']} won. "
        f"Top scorer: {row['top_scorer']} "
        f"with {row['top_score']} runs."
    )

    sentences.append(sentence)


print("Created match descriptions")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded")

embeddings = model.encode(
    sentences
)
print("Embeddings created")
client = chromadb.Client()

collection = client.create_collection(
    name="cricket_matches"
)

collection.add(
    ids=[str(i) for i in range(len(sentences))],
    documents=sentences,
    embeddings=embeddings.tolist(),
)

print("Vectors stored in ChromaDB")

def search_matches(query):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
    )
    return results


while True:
    query = input(
        "\nAsk a question (or type exit): "
    )
    if query.lower() == "exit":
        break

    results = search_matches(query)
    print("\nTop Matches:\n")
    for match in results["documents"][0]:
        print(match)
        print("-" * 50)