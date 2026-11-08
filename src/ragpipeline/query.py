"""Find the paragraphs closest to a query in the ChromaDB collection built by main.py."""

import argparse
import textwrap

import chromadb
import numpy as np
from chromadb.errors import NotFoundError
from FlagEmbedding import BGEM3FlagModel

from ragpipeline.main import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PCA_FILE,
    l2_normalize,
)


def embed_query(query: str, model: BGEM3FlagModel, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Run the query through the same steps as the corpus: embed, L2, PCA, L2."""
    dense = model.encode_queries([query])["dense_vecs"]
    projected = (l2_normalize(dense) - mean) @ components.T
    return l2_normalize(projected)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search The Brothers Karamazov by meaning.")
    parser.add_argument("query", help="the text to search for")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="number of paragraphs to return")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        raise SystemExit(f"No collection {COLLECTION_NAME!r} in {CHROMA_PATH}; run main.py first")
    pca = np.load(PCA_FILE)

    model = BGEM3FlagModel(EMBEDDING_MODEL, normalize_embeddings=False, use_fp16=True)
    query_vector = embed_query(args.query, model, pca["mean"], pca["components"])

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    for rank, (document, metadata, distance) in enumerate(hits, start=1):
        # The collection uses cosine distance, so 1 - distance is the cosine similarity.
        print(f"\n#{rank}  paragraph {metadata['paragraph']}  score {1 - distance:.3f}")
        print(textwrap.fill(document, width=100))


if __name__ == "__main__":
    main()
