"""Split a text into paragraphs, embed them with BGE-M3, reduce them with PCA and store them in ChromaDB."""

import re
from pathlib import Path

import chromadb
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.decomposition import PCA

# BAAI/bge-reranker-v2-m3 is a cross-encoder: it scores (query, passage) pairs and
# cannot produce embeddings. BAAI/bge-m3 is the embedding model it was built on.
EMBEDDING_MODEL = "BAAI/bge-m3"
MAX_LENGTH = 8192  # bge-m3's context window; long enough that no paragraph is truncated
PCA_COMPONENTS = 256  # down from 1024; keeps ~90% of the variance on this corpus

SOURCE_TEXT = Path("brothers-kamarazov.txt")
CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "brothers-karamazov"
# Queries must go through the same projection: l2_normalize((q - mean) @ components.T)
PCA_FILE = Path("brothers-kamarazov-pca.npz")


def load_paragraphs(path: Path) -> list[str]:
    """Split on blank lines and unwrap the hard line breaks inside each paragraph."""
    text = path.read_text(encoding="utf-8")
    paragraphs = re.split(r"\n\s*\n", text)
    return [" ".join(p.split()) for p in paragraphs if p.strip()]


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit L2 norm."""
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def main() -> None:
    paragraphs = load_paragraphs(SOURCE_TEXT)
    print(f"Loaded {len(paragraphs)} paragraphs from {SOURCE_TEXT}")

    model = BGEM3FlagModel(EMBEDDING_MODEL, normalize_embeddings=False, use_fp16=True)
    output = model.encode_corpus(paragraphs, batch_size=32, max_length=MAX_LENGTH)
    embeddings = l2_normalize(output["dense_vecs"])
    print(f"Embeddings shape: {embeddings.shape}")

    pca = PCA(n_components=PCA_COMPONENTS)
    embeddings = l2_normalize(pca.fit_transform(embeddings))

    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"After PCA: shape {embeddings.shape}, "
        f"explained variance {pca.explained_variance_ratio_.sum():.1%}, "
        f"L2 norms in [{norms.min():.6f}, {norms.max():.6f}]"
    )

    np.savez(PCA_FILE, mean=pca.mean_, components=pca.components_)
    print(f"Saved PCA projection to {PCA_FILE}")

    # Rebuild the collection on every run so it always matches the PCA saved above.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    # The vectors come from bge-m3 + PCA, so Chroma must not embed anything itself.
    collection = client.create_collection(
        COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )

    batch_size = client.get_max_batch_size()
    for start in range(0, len(paragraphs), batch_size):
        indices = range(start, min(start + batch_size, len(paragraphs)))
        collection.add(
            ids=[f"paragraph-{i}" for i in indices],
            embeddings=embeddings[indices.start : indices.stop],
            documents=paragraphs[indices.start : indices.stop],
            metadatas=[{"paragraph": i} for i in indices],
        )
    print(f"Stored {collection.count()} paragraphs in collection {COLLECTION_NAME!r} at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
