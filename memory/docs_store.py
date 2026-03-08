"""
Static KB / docs vector store backed by ChromaDB.
Indexes HR and IT documents for retrieval.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from utils.config import get_config, DATA_DIR, CHROMA_DOCS_DIR
from utils.embeddings import embed_texts

_cfg = get_config().retrieval


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks by character count."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]


class DocsStore:
    """ChromaDB-backed store for static KB articles."""

    def __init__(self, persist_dir: str = CHROMA_DOCS_DIR):
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=persist_dir,
        ))
        self.collection = self.client.get_or_create_collection(
            name="kb_docs",
            metadata={"hnsw:space": "cosine"},
        )

    # ── Indexing ───────────────────────────────────────────────────────────
    def index_documents(self, force: bool = False) -> int:
        """Read hr_docs.txt and it_docs.txt, chunk, embed, and store.
        Returns number of chunks indexed.
        """
        if self.collection.count() > 0 and not force:
            return self.collection.count()

        docs_files = {
            "HR": DATA_DIR / "hr_docs.txt",
            "IT": DATA_DIR / "it_docs.txt",
        }

        all_chunks: List[str] = []
        all_ids: List[str] = []
        all_meta: List[Dict[str, Any]] = []

        for source, fpath in docs_files.items():
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{source}_{i}_{chunk[:50]}".encode()).hexdigest()
                all_chunks.append(chunk)
                all_ids.append(doc_id)
                all_meta.append({"source": source, "chunk_index": i})

        if not all_chunks:
            return 0

        embeddings = embed_texts(all_chunks)
        self.collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_meta,
        )
        return len(all_chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Return the top-k most relevant doc chunks for a query."""
        top_k = top_k or _cfg.docs_top_k
        if self.collection.count() == 0:
            return []

        q_emb = embed_texts([query])
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist  # ChromaDB cosine distance → similarity
            if similarity >= _cfg.similarity_threshold:
                hits.append({
                    "text": doc,
                    "source": meta.get("source", ""),
                    "similarity": round(similarity, 4),
                })
        return hits
