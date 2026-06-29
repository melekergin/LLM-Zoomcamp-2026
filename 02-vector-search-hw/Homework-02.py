from embedder import Embedder

embedder = Embedder("models/Xenova/all-MiniLM-L6-v2")

query = "How does approximate nearest neighbor search work?"
v = embedder.encode(query)

print(len(v))   # sanity check — should be 384
print(v[0])     # the answer to Q1 : 
# 384
# -0.02058203437252893

# Q2 cosine similarity between two vectors

import requests

url = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/8c1834d/02-vector-search/lessons/07-sqlitesearch-vector.md"
content = requests.get(url).text

v_page = embedder.encode(content)

import numpy as np
similarity = np.dot(v, v_page)
print(similarity)

#Q3 load all 72 pages, chunk them, and find which chunk scores highest against your Q1 query
from gitsource import GithubRepositoryDataReader, chunk_documents

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

chunks = chunk_documents(documents, size=2000, step=1000)

vectors = [embedder.encode(c["content"]) for c in chunks]
X = np.array(vectors)

scores = X.dot(v)
best_idx = scores.argmax()
print(chunks[best_idx]["filename"])

#Q4  — vector search with minsearch's VectorSearch

from minsearch import VectorSearch

vs = VectorSearch()
vs.fit(X, chunks)   # X = your chunk vectors, chunks = the matching metadata

query2 = "What metric do we use to evaluate a search engine?"
v2 = embedder.encode(query2)

results = vs.search(v2, num_results=5)
print(results[0]["filename"])  # 04-evaluation/lessons/05-search-metrics.md



#Q5 vector search vs. keyword search
from minsearch import Index

text_index = Index(text_fields=["content"])
text_index.fit(chunks)

query3 = "How do I store vectors in PostgreSQL?"
v3 = embedder.encode(query3)

vector_results = vs.search(v3, num_results=5)
text_results = text_index.search(query3, num_results=5)

vector_files = {r["filename"] for r in vector_results}
text_files = {r["filename"] for r in text_results}

print(vector_files - text_files)



# Q6 — hybrid search with RRF.
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

query4 = "How do I give the model access to tools?"
v4 = embedder.encode(query4)

vector_results4 = vs.search(v4, num_results=5)
text_results4 = text_index.search(query4, num_results=5)

combined = rrf([vector_results4, text_results4])
print(combined[0]["filename"])