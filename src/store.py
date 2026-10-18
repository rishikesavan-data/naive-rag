
from pathlib import Path

import chromadb
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = BASE_DIR / "data" / "embedded" / "embeddings.npz"
CHUNKS_PATH = BASE_DIR / "data" / "extracted" / "chunks.jsonl"
DB_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "ai_ml_knowledge"


# ============================================================
# 1. LOAD EMBEDDINGS AND CHUNKS
# ============================================================

def load_data():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}"
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with np.load(EMBEDDINGS_PATH, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        embedding_ids = data["ids"].tolist()

    # Load authoritative text and metadata from chunks.jsonl.
    import json

    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but "
            f"{len(embeddings)} embeddings."
        )

    chunk_ids = [chunk["id"] for chunk in chunks]

    if chunk_ids != embedding_ids:
        raise ValueError(
            "Chunk IDs in embeddings.npz do not match chunks.jsonl. "
            "Run src/embed.py again after regenerating chunks."
        )

    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        raise ValueError(
            f"Unexpected embedding dimensions: {embeddings.shape}"
        )

    return chunks, embeddings


# ============================================================
# 2. CREATE PERSISTENT CHROMADB COLLECTION
# ============================================================

def main():
    print("Loading chunks and embeddings...")

    chunks, embeddings = load_data()

    print(f"Chunks loaded: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")

    DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Recreate the collection to prevent stale records or duplicates
    # when this script is rerun.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Existing collection deleted.")
    except Exception:
        print("No existing collection found; creating a new one.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB accepts batches. This avoids sending all records
    # in one potentially large request.
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end]

        collection.add(
            ids=[chunk["id"] for chunk in batch_chunks],
            embeddings=batch_embeddings.tolist(),
            documents=[chunk["text"] for chunk in batch_chunks],
            metadatas=[
                {
                    "source": chunk["source"],
                    "heading": chunk["heading"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "character_count": int(chunk["character_count"]),
                }
                for chunk in batch_chunks
            ],
        )

        print(f"Stored records: {end}/{len(chunks)}")

    # ========================================================
    # 3. VERIFY DATABASE
    # ========================================================

    stored_count = collection.count()

    print("\n========== CHROMADB RESULTS ==========")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database path: {DB_PATH}")
    print(f"Expected records: {len(chunks)}")
    print(f"Stored records: {stored_count}")

    if stored_count != len(chunks):
        raise RuntimeError("Record count verification failed.")

    sample = collection.peek(limit=1)

    if sample["ids"]:
        print(f"Sample ID: {sample['ids'][0]}")
        print(f"Sample metadata: {sample['metadatas'][0]}")
        print(f"Sample document: {sample['documents'][0][:200]}")

    print("\nVector database storage completed successfully.")


if __name__ == "__main__":
    main()
