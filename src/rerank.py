
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "ai_ml_knowledge"

# Model used to retrieve candidate chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model used to rerank candidate chunks
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVAL = 10
TOP_K_RERANKED = 5


def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading reranking model...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Database records: {collection.count()}")
    print("Reranking system is ready.")

    while True:
        question = input(
            "\nEnter your question (or type 'exit' to stop): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Reranking test ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # 1. Embed the question
        query_embedding = embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # 2. Retrieve candidate chunks
        candidate_count = min(
            TOP_K_RETRIEVAL,
            collection.count(),
        )

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not documents:
            print("No relevant chunks found.")
            continue

        # 3. Score each query-document pair
        pairs = [[question, document] for document in documents]
        scores = reranker.predict(pairs)

        # 4. Sort by cross-encoder score, highest first
        ranked_results = sorted(
            zip(scores, documents, metadatas, distances),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        # 5. Display the best five chunks
        print("\n========== RERANKED RESULTS ==========")

        for rank, (score, document, metadata, distance) in enumerate(
            ranked_results[:TOP_K_RERANKED],
            start=1,
        ):
            print(f"\n--- Result {rank} ---")
            print(f"Heading: {metadata['heading']}")
            print(f"Source: {metadata['source']}")
            print(f"Chunk index: {metadata['chunk_index']}")
            print(f"Reranker score: {float(score):.4f}")
            print(f"Original cosine distance: {distance:.4f}")
            print(f"\n{document[:1000]}")


if __name__ == "__main__":
    main()
