
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "ai_ml_knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def main():
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Database records: {collection.count()}")

    while True:
        question = input(
            "\nEnter your question (or type 'exit' to stop): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Retrieval test ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        query_embedding = model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        print("\n========== RETRIEVAL RESULTS ==========")

        for rank, (document, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
            print(f"\n--- Result {rank} ---")
            print(f"Heading: {metadata['heading']}")
            print(f"Source: {metadata['source']}")
            print(f"Chunk index: {metadata['chunk_index']}")
            print(f"Cosine distance: {distance:.4f}")
            print(f"\n{document[:1000]}")

    print("\nRetrieval session finished.")


if __name__ == "__main__":
    main()
