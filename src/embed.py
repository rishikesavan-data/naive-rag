
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "extracted" / "chunks.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "embedded" / "embeddings.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if not chunks:
        raise ValueError("No chunks found.")

    return chunks


def main():
    print("Loading chunks...")
    chunks = load_chunks()

    texts = [chunk["text"] for chunk in chunks]

    print(f"Chunks loaded: {len(texts)}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(chunks), 384):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        embeddings=embeddings,
        ids=np.array([chunk["id"] for chunk in chunks]),
        texts=np.array(texts),
        headings=np.array([chunk["heading"] for chunk in chunks]),
        sources=np.array([chunk["source"] for chunk in chunks]),
        chunk_indices=np.array(
            [chunk["chunk_index"] for chunk in chunks],
            dtype=np.int32,
        ),
    )

    print("\n========== EMBEDDING RESULTS ==========")
    print(f"Number of vectors: {embeddings.shape[0]}")
    print(f"Dimensions per vector: {embeddings.shape[1]}")
    print(f"Data type: {embeddings.dtype}")
    print(f"Normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
