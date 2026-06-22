# 01-intro — Building a RAG Pipeline From Scratch

Part of [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp). This module builds a working Retrieval-Augmented Generation (RAG) system over a course FAQ dataset, starting from plain keyword search and ending with a persistent, modular architecture.

## What this module covers

- Keyword search vs. vector search — and why RAG doesn't require embeddings
- Building a search → prompt → LLM pipeline from individual functions
- Wrapping the pipeline in a reusable class (`RAGBase`)
- Separating data ingestion from querying with a persistent index

## Architecture

```
User Question
     │
     ▼
  search()  ──────────────►  Search Index (minsearch / sqlitesearch)
     │
     ▼
build_prompt()  ─── packages retrieved docs + question into a prompt
     │
     ▼
   llm()  ──────────────►  OpenAI Responses API
     │
     ▼
   Answer
```

With persistent ingestion, this splits into two independent processes:

```
INGESTION (runs once)              QUERY (runs every time)
────────────────────                ────────────────────
FAQ data → parse → index    ──►     open faq.db → search → RAG pipeline
   → write to faq.db
```

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Loads the FAQ dataset (`load_faq_data`) and builds a search index (`build_index`) |
| `rag_helper.py` | `RAGBase` class — encapsulates search, prompt building, and LLM calls into one reusable object |
| `start.ipynb` | Main notebook — wires up `ingest.py` + `rag_helper.py` and runs the RAG pipeline interactively |
| `sqlite-ingest.ipynb` | Ingestion-only notebook — writes a persistent index to `faq.db` using `sqlitesearch` |
| `persinsent_rag.ipynb` | Query-only notebook — connects to the existing `faq.db` and runs RAG against it |

## Key concepts

### Keyword search vs. vector search

`minsearch` retrieves documents by matching **words**, not meaning:

```
text → index words → exact / partial word match
```

Vector search instead converts text into embeddings and compares **meaning**:

```
text → embedding model → vector → distance comparison
```

This is why a query like *"is it too late to join?"* matches an FAQ entry phrased *"can I still join?"* under vector search but may miss it entirely under pure keyword search — there's almost no word overlap, but the meaning is identical.

### `boost_dict` vs. `filter_dict`

```python
search_results = index.search(
    query,
    boost_dict={"question": 2.0, "section": 0.5},  # changes ranking
    filter_dict={"course": "llm-zoomcamp"},          # changes eligibility
    num_results=5
)
```

- **`boost_dict`** weights how strongly a field match affects the relevance score.
- **`filter_dict`** is a hard exclusion — equivalent to a SQL `WHERE` clause. No ranking can override it.

### The `RAGBase` class

Wrapping the pipeline in a class solves a real problem: running multiple RAG systems (e.g. one per course) without their settings (`course`, `model`, `index`) colliding via shared global variables.

```python
llm_rag = RAGBase(index=llm_index, llm_client=client, course="llm-zoomcamp")
de_rag  = RAGBase(index=de_index,  llm_client=client, course="data-engineering-zoomcamp")
```

Each instance keeps its own `self.course`, `self.index`, and `self.model` — fully isolated from any other instance built from the same class.

### Persistent ingestion

`minsearch` is in-memory — close the process, the index is gone, and you re-index on every restart. Fine for a small FAQ dataset; not viable for large or slow-to-build datasets.

`sqlitesearch` is a drop-in replacement with the same API, backed by a SQLite file (`faq.db`) on disk:

```python
from sqlitesearch import TextSearchIndex

index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db"
)
```

One process ingests data into `faq.db` once; any number of other processes can query it afterward without re-running ingestion. This mirrors a standard data warehouse pattern: an ETL job populates a persistent store once, and downstream reports read from it repeatedly without re-running the extraction.

## Setup

```bash
pip install minsearch sqlitesearch python-dotenv openai
```

Set your OpenAI API key as an environment variable (don't hardcode it):

```bash
export OPENAI_API_KEY="sk-..."
```

Or use a `.env` file with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Running it

```python
from ingest import load_faq_data, build_index
from rag_helper import RAGBase
from openai import OpenAI

documents = load_faq_data()
index = build_index(documents)

openai_client = OpenAI()

rag_system = RAGBase(index=index, llm_client=openai_client, course="llm-zoomcamp")

answer = rag_system.rag("is it too late to join the course?")
print(answer)
```

## What's next

Part 2 of this module moves from a fixed pipeline to **agentic RAG** — where the LLM decides when and what to search, rather than running the same retrieve-then-generate sequence on every query.