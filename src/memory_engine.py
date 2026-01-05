from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from datetime import datetime
from memory_utils import days_since, decay


MEMORY_DIR = Path("./memory/chroma_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="kitai_memory",
    persist_directory=str(MEMORY_DIR),
    embedding_function=embeddings
)

def downgrade_old_memories(mem_type: str, keyword: str, decay_to: float=0.3):
    results = db.get()
    for i, meta in enumerate(results["metadatas"]):
        if meta.get("type") == mem_type:
            meta["confidence"] = decay_to

def delete_by_key(key):
    results = db.get(where={"key": key})
    if results and results["ids"]:
        db.delete(ids=results["ids"])

def store_memory(text: str, mem_type: str, keyword: str = '', confidence: float = 1.0):
    if keyword:
        delete_by_key(keyword)

    db.add_texts(
        texts=[text],
        metadatas=[{
            "type": mem_type,
            "confidence": confidence,
            "key": keyword,
            "timestamp": datetime.utcnow().isoformat()
        }]
    )


def retrieve_memory(query: str, k: int = 8):
    results = db.similarity_search(query, k=k)
    strong = []

    for m in results:
        ts = m.metadata.get("timestamp")
        conf = m.metadata.get("confidence", 0)

        if not ts:
            continue

        effective_conf = decay(conf, days_since(ts))

        if effective_conf >= 0.5:
            m.metadata["effective_conf"] = round(effective_conf, 2)
            strong.append(m)

    return strong

def retrieve_strong_memory(query: str, threshold=0.5):
    results = retrieve_memory(query)
    return [
        m for m in results
        if m.metadata.get("confidence", 0) >= threshold
    ]


