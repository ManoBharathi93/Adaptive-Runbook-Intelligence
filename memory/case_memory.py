"""
Episodic Case Memory – stores past ticket resolutions with reward signals.
Supports reward‑weighted retrieval so higher‑reward cases rank higher.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from utils.config import get_config, DATA_DIR, CHROMA_CASES_DIR
from utils.embeddings import embed_texts

_cfg = get_config().retrieval

# ── Pydantic model for a case ─────────────────────────────────────────────────
from pydantic import BaseModel, Field


class CaseRecord(BaseModel):
    ticket_id: str
    problem: str
    context: Dict[str, Any] = {}
    actions_taken: List[str] = []
    outcome: str = "pending"
    escalation_level: int = 0
    reopen_count: int = 0
    reward_score: float = 0.0
    query: str = ""


class CaseMemory:
    """ChromaDB‑backed episodic memory for past resolved tickets."""

    def __init__(self, persist_dir: str = CHROMA_CASES_DIR):
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=persist_dir,
        ))
        self.collection = self.client.get_or_create_collection(
            name="case_memory",
            metadata={"hnsw:space": "cosine"},
        )

    # ── Seed from synthetic data ──────────────────────────────────────────
    def seed_from_json(self, path: Optional[Path] = None, force: bool = False) -> int:
        """Load synthetic_tickets.json into ChromaDB. Returns count of records."""
        if self.collection.count() > 0 and not force:
            return self.collection.count()

        path = path or (DATA_DIR / "synthetic_tickets.json")
        if not path.exists():
            return 0

        with open(path, encoding="utf-8") as f:
            tickets = json.load(f)

        records = [CaseRecord(**t) for t in tickets]
        return self._upsert_records(records)

    # ── Insert / update ──────────────────────────────────────────────────
    def add_case(self, case: CaseRecord) -> None:
        self._upsert_records([case])

    def _upsert_records(self, records: List[CaseRecord]) -> int:
        if not records:
            return 0

        texts = [self._case_to_text(r) for r in records]
        ids = [r.ticket_id for r in records]
        embeddings = embed_texts(texts)
        metadatas = [self._case_to_meta(r) for r in records]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(records)

    # ── Retrieval (reward‑weighted) ───────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int | None = None,
        reward_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past cases.  Final ranking:
            score = similarity * (1 - reward_weight) + normalised_reward * reward_weight
        This biases retrieval toward cases that had positive outcomes.
        """
        top_k = top_k or _cfg.cases_top_k
        if self.collection.count() == 0:
            return []

        fetch_k = min(top_k * 3, self.collection.count())  # over‑fetch then re‑rank
        q_emb = embed_texts([query])
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist
            if similarity < _cfg.similarity_threshold:
                continue

            reward = float(meta.get("reward_score", 0.0))
            norm_reward = (reward + 1.0) / 2.0  # map [-1, 1] → [0, 1]
            combined = similarity * (1 - reward_weight) + norm_reward * reward_weight

            hits.append({
                "ticket_id": meta.get("ticket_id", ""),
                "problem": meta.get("problem", ""),
                "actions_taken": meta.get("actions_taken", ""),
                "outcome": meta.get("outcome", ""),
                "escalation_level": int(meta.get("escalation_level", 0)),
                "reopen_count": int(meta.get("reopen_count", 0)),
                "reward_score": reward,
                "similarity": round(similarity, 4),
                "combined_score": round(combined, 4),
                "text": doc,
            })

        hits.sort(key=lambda h: h["combined_score"], reverse=True)
        return hits[:top_k]

    # ── Update reward for existing case ───────────────────────────────────
    def update_reward(self, ticket_id: str, new_reward: float) -> bool:
        """Update the reward_score for an existing case and re-embed."""
        try:
            result = self.collection.get(ids=[ticket_id], include=["metadatas", "documents"])
            if not result["ids"]:
                return False
            meta = result["metadatas"][0]
            meta["reward_score"] = new_reward
            doc = result["documents"][0]
            embedding = embed_texts([doc])
            self.collection.update(
                ids=[ticket_id],
                documents=[doc],
                embeddings=embedding,
                metadatas=[meta],
            )
            return True
        except Exception:
            return False

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _case_to_text(r: CaseRecord) -> str:
        return (
            f"Problem: {r.problem}\n"
            f"Query: {r.query}\n"
            f"Context: {json.dumps(r.context)}\n"
            f"Actions: {', '.join(r.actions_taken)}\n"
            f"Outcome: {r.outcome}"
        )

    @staticmethod
    def _case_to_meta(r: CaseRecord) -> Dict[str, Any]:
        return {
            "ticket_id": r.ticket_id,
            "problem": r.problem,
            "actions_taken": ", ".join(r.actions_taken),
            "outcome": r.outcome,
            "escalation_level": r.escalation_level,
            "reopen_count": r.reopen_count,
            "reward_score": r.reward_score,
        }
