from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from datetime import datetime

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


def store_memory(text: str, mem_type: str, keyword: str, confidence: float = 1.0):
    # Downgrade old memories of same type
    if mem_type in ["fact", "preference", "identity", "goal"]:
        downgrade_old_memories(mem_type, keyword)

    db.add_texts(
        texts=[text],
        metadatas=[{
            "type": mem_type,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }]
    )


def retrieve_memory(query: str, k: int = 5):
    return db.similarity_search(query, k=k)

def retrieve_strong_memory(query: str, threshold=0.5):
    results = retrieve_memory(query)
    return [
        m for m in results
        if m.metadata.get("confidence", 0) >= threshold
    ]


