"""
Runbook Intelligence Store – the core artifact of the Runbook Intelligence Platform.

Stores, retrieves, promotes, and manages runbooks — known-good and known-bad
automation sequences discovered from real execution data.

Persistence:
    ChromaDB  → trigger_embedding (vector similarity lookup)
    SQLite    → structured counters (success, failure, reopen, latency, tokens)

Promotion logic:
    EXPERIMENTAL  → runs < promotion_threshold
    KNOWN_GOOD    → success_count ≥ threshold AND failure_rate < 20%
    KNOWN_BAD     → failure_count ≥ bad_threshold OR reopen_rate > 50%
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

from utils.config import get_config, DB_DIR
from utils.embeddings import embed_texts

# ── Constants ─────────────────────────────────────────────────────────────────
RUNBOOK_CHROMA_DIR = str(DB_DIR / "chroma_runbooks")
RUNBOOK_SQLITE_PATH = DB_DIR / "runbooks.db"

PROMOTION_THRESHOLD = 3       # min successes → KNOWN_GOOD
KNOWN_BAD_THRESHOLD = 3       # min failures  → KNOWN_BAD
REOPEN_BAD_RATIO = 0.50       # reopen_count / total > this → KNOWN_BAD


# ═══════════════════════════════════════════════════════════════════════════════
# Runbook Schema (Pydantic)
# ═══════════════════════════════════════════════════════════════════════════════

class RunbookStatus:
    EXPERIMENTAL = "EXPERIMENTAL"
    KNOWN_GOOD   = "KNOWN_GOOD"
    KNOWN_BAD    = "KNOWN_BAD"


class Runbook(BaseModel):
    runbook_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger_text: str = ""
    steps: List[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    reopen_count: int = 0
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    risk_level: str = "MEDIUM"
    status: str = RunbookStatus.EXPERIMENTAL
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_used_at: str = ""

    @property
    def total_uses(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses

    @property
    def reopen_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.reopen_count / self.total_uses

    def compute_status(self) -> str:
        """Re-evaluate status based on current counters."""
        if self.failure_count >= KNOWN_BAD_THRESHOLD:
            return RunbookStatus.KNOWN_BAD
        if self.total_uses > 0 and self.reopen_rate > REOPEN_BAD_RATIO:
            return RunbookStatus.KNOWN_BAD
        if self.success_count >= PROMOTION_THRESHOLD and self.success_rate >= 0.80:
            return RunbookStatus.KNOWN_GOOD
        return RunbookStatus.EXPERIMENTAL


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite schema
# ═══════════════════════════════════════════════════════════════════════════════

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runbooks (
    runbook_id      TEXT PRIMARY KEY,
    trigger_text    TEXT NOT NULL,
    steps           TEXT NOT NULL,
    success_count   INTEGER DEFAULT 0,
    failure_count   INTEGER DEFAULT 0,
    reopen_count    INTEGER DEFAULT 0,
    avg_latency_ms  REAL    DEFAULT 0,
    avg_tokens      REAL    DEFAULT 0,
    risk_level      TEXT    DEFAULT 'MEDIUM',
    status          TEXT    DEFAULT 'EXPERIMENTAL',
    created_at      TEXT,
    last_used_at    TEXT
)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# RunbookStore
# ═══════════════════════════════════════════════════════════════════════════════

class RunbookStore:
    """
    Dual-backend runbook store:
        ChromaDB  → trigger-text embeddings for similarity search
        SQLite    → structured counters and metadata
    """

    def __init__(
        self,
        chroma_dir: str = RUNBOOK_CHROMA_DIR,
        sqlite_path: Path | str = RUNBOOK_SQLITE_PATH,
    ):
        DB_DIR.mkdir(parents=True, exist_ok=True)

        # ChromaDB for embeddings
        self.chroma = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=chroma_dir,
        ))
        self.collection = self.chroma.get_or_create_collection(
            name="runbooks",
            metadata={"hnsw:space": "cosine"},
        )

        # SQLite for structured data
        self.db = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(_CREATE_TABLE)
        self.db.commit()

        self._chroma_dir = chroma_dir
        self._sqlite_path = sqlite_path

    # ── Create / upsert ──────────────────────────────────────────────────
    def create_runbook(self, trigger_text: str, steps: List[str],
                       risk_level: str = "MEDIUM",
                       latency_ms: float = 0, tokens: float = 0) -> Runbook:
        """Create or strengthen an EXPERIMENTAL runbook from exploratory execution."""
        # Reuse an existing close match with identical steps so repeated tickets
        # strengthen one runbook instead of creating duplicates.
        existing = self.find_matching_runbook(trigger_text, min_similarity=0.90, exclude_bad=True)
        if existing:
            candidate = existing["runbook"]
            if candidate.steps == steps:
                updated = self.record_execution(
                    runbook_id=candidate.runbook_id,
                    success=True,
                    latency_ms=latency_ms,
                    tokens=tokens,
                    reopened=False,
                )
                if updated is not None:
                    return updated

        rb = Runbook(
            trigger_text=trigger_text,
            steps=steps,
            risk_level=risk_level.upper(),
            avg_latency_ms=latency_ms,
            avg_tokens=tokens,
            success_count=1,  # created from a successful execution
        )
        self._persist(rb)
        return rb

    def _persist(self, rb: Runbook) -> None:
        """Upsert into both ChromaDB and SQLite."""
        # ChromaDB: store trigger embedding
        emb = embed_texts([rb.trigger_text])
        self.collection.upsert(
            ids=[rb.runbook_id],
            documents=[rb.trigger_text],
            embeddings=emb,
            metadatas=[{"runbook_id": rb.runbook_id, "status": rb.status}],
        )

        # SQLite: upsert structured data
        self.db.execute("""
            INSERT INTO runbooks
                (runbook_id, trigger_text, steps, success_count, failure_count,
                 reopen_count, avg_latency_ms, avg_tokens, risk_level, status,
                 created_at, last_used_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(runbook_id) DO UPDATE SET
                success_count  = excluded.success_count,
                failure_count  = excluded.failure_count,
                reopen_count   = excluded.reopen_count,
                avg_latency_ms = excluded.avg_latency_ms,
                avg_tokens     = excluded.avg_tokens,
                risk_level     = excluded.risk_level,
                status         = excluded.status,
                last_used_at   = excluded.last_used_at
        """, (
            rb.runbook_id, rb.trigger_text, json.dumps(rb.steps),
            rb.success_count, rb.failure_count, rb.reopen_count,
            rb.avg_latency_ms, rb.avg_tokens, rb.risk_level,
            rb.status, rb.created_at, rb.last_used_at,
        ))
        self.db.commit()

    # ── Retrieval ─────────────────────────────────────────────────────────
    def find_matching_runbook(
        self,
        query: str,
        min_similarity: float = 0.75,
        exclude_bad: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best matching runbook for a query.
        Returns None if no runbook matches above min_similarity.
        Only returns KNOWN_GOOD or EXPERIMENTAL (never KNOWN_BAD if exclude_bad).
        """
        if self.collection.count() == 0:
            return None

        q_emb = embed_texts([query])
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=min(3, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        candidates: List[Dict[str, Any]] = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist
            if similarity < min_similarity:
                continue

            rb_id = meta.get("runbook_id", "")
            rb = self.get_runbook(rb_id)
            if rb is None:
                continue

            if exclude_bad and rb.status == RunbookStatus.KNOWN_BAD:
                continue

            candidates.append({
                "runbook": rb,
                "similarity": round(similarity, 4),
                "trigger_text": doc,
            })

        if not candidates:
            return None

        # Prefer KNOWN_GOOD first, then higher similarity, then more total successful evidence.
        status_rank = {
            RunbookStatus.KNOWN_GOOD: 2,
            RunbookStatus.EXPERIMENTAL: 1,
            RunbookStatus.KNOWN_BAD: 0,
        }

        candidates.sort(
            key=lambda item: (
                status_rank.get(item["runbook"].status, 0),
                item["similarity"],
                item["runbook"].success_count,
                item["runbook"].total_uses,
            ),
            reverse=True,
        )

        return candidates[0]

    def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """Load a runbook from SQLite by ID."""
        row = self.db.execute(
            "SELECT * FROM runbooks WHERE runbook_id = ?", (runbook_id,)
        ).fetchone()
        if row is None:
            return None
        return Runbook(
            runbook_id=row["runbook_id"],
            trigger_text=row["trigger_text"],
            steps=json.loads(row["steps"]),
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            reopen_count=row["reopen_count"],
            avg_latency_ms=row["avg_latency_ms"],
            avg_tokens=row["avg_tokens"],
            risk_level=row["risk_level"],
            status=row["status"],
            created_at=row["created_at"] or "",
            last_used_at=row["last_used_at"] or "",
        )

    # ── Update after execution ─────────────────────────────────────────────
    def record_execution(
        self,
        runbook_id: str,
        success: bool,
        latency_ms: float = 0,
        tokens: float = 0,
        reopened: bool = False,
    ) -> Optional[Runbook]:
        """Update counters after a runbook execution, recompute status."""
        rb = self.get_runbook(runbook_id)
        if rb is None:
            return None

        if success:
            rb.success_count += 1
        else:
            rb.failure_count += 1

        if reopened:
            rb.reopen_count += 1

        # Running average for latency and tokens
        total = rb.total_uses
        if total > 0:
            rb.avg_latency_ms = (
                (rb.avg_latency_ms * (total - 1) + latency_ms) / total
            )
            rb.avg_tokens = (
                (rb.avg_tokens * (total - 1) + tokens) / total
            )

        rb.last_used_at = datetime.now(timezone.utc).isoformat()
        rb.status = rb.compute_status()

        self._persist(rb)
        return rb

    # ── Bulk queries ──────────────────────────────────────────────────────
    def list_all(self) -> List[Runbook]:
        """Return all runbooks from SQLite."""
        rows = self.db.execute("SELECT * FROM runbooks ORDER BY created_at DESC").fetchall()
        return [
            Runbook(
                runbook_id=r["runbook_id"],
                trigger_text=r["trigger_text"],
                steps=json.loads(r["steps"]),
                success_count=r["success_count"],
                failure_count=r["failure_count"],
                reopen_count=r["reopen_count"],
                avg_latency_ms=r["avg_latency_ms"],
                avg_tokens=r["avg_tokens"],
                risk_level=r["risk_level"],
                status=r["status"],
                created_at=r["created_at"] or "",
                last_used_at=r["last_used_at"] or "",
            )
            for r in rows
        ]

    def count_by_status(self) -> Dict[str, int]:
        """Count runbooks by status."""
        rows = self.db.execute(
            "SELECT status, COUNT(*) as cnt FROM runbooks GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def count_known_bad_avoided(self) -> int:
        """Count how many KNOWN_BAD runbooks exist (each represents an avoided path)."""
        row = self.db.execute(
            "SELECT COUNT(*) as cnt FROM runbooks WHERE status = 'KNOWN_BAD'"
        ).fetchone()
        return row["cnt"] if row else 0

    def total_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM runbooks").fetchone()[0]
