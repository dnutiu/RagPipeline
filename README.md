# RagPipeline

Semantic search over *The Brothers Karamazov*. The book is split into paragraphs, embedded with
[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) and stored in a local [ChromaDB](https://www.trychroma.com/)
database that you can query in plain language.

## Setup

Requires [uv](https://docs.astral.sh/uv/); it installs Python 3.14 and the dependencies for you.

```bash
uv sync
```

## Usage

Run both commands from the project root.

Build the database (the first run downloads the ~2.3 GB model):

```bash
uv run src/ragpipeline/main.py
```

Search it:

```bash
uv run src/ragpipeline/query.py "Who killed Fyodor Pavlovitch?" -k 3
```

`-k` sets how many paragraphs to return (default 5).

## How it works

`main.py`:

1. Splits `brothers-kamarazov.txt` on blank lines, one entry per paragraph.
2. Embeds each paragraph with bge-m3 (1024 dimensions) and L2-normalizes it.
3. Reduces the vectors to 256 dimensions with PCA and L2-normalizes them again.
4. Stores the vectors, paragraph text and paragraph numbers in `chroma_db/`, using cosine distance,
   and saves the PCA projection to `brothers-kamarazov-pca.npz`.

`query.py` runs the query through the same steps (embed, normalize, PCA, normalize), asks ChromaDB
for the nearest paragraphs and prints them with their cosine similarity.

Rerun `main.py` whenever you change the text or `PCA_COMPONENTS`; it rebuilds the collection from scratch.

## Credits

The text of *The Brothers Karamazov* by Fyodor Dostoyevsky (translated by Constance Garnett) comes from
[Project Gutenberg](https://www.gutenberg.org/ebooks/28054). It is used under the
[Project Gutenberg License](https://www.gutenberg.org/policy/license.html).
