"""
Feedback Collector – persists feedback signals to SQLite and
propagates computed rewards back into Case Memory.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from utils.config import SQLITE_PATH, DB_DIR
from policy.reward_calculator import RewardCalculator, FeedbackSignals
from memory.case_memory import CaseMemory


def _ensure_db(db_path: Path = SQLITE_PATH) -> sqlite3.Connection:
    """Create the feedback table if it doesn't exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id       TEXT NOT NULL,
            ticket_reopened INTEGER DEFAULT 0,
            analyst_override INTEGER DEFAULT 0,
            user_rating     INTEGER DEFAULT 4,
            reward_score    REAL,
            created_at      TEXT,
            metadata        TEXT
        )
    """)
    conn.commit()
    return conn


class FeedbackCollector:
    """Collects simulated delayed feedback, computes rewards, and updates case memory."""

    def __init__(
        self,
        db_path: Path = SQLITE_PATH,
        case_memory: Optional[CaseMemory] = None,
    ):
        self.conn = _ensure_db(db_path)
        self.reward_calc = RewardCalculator()
        self.case_memory = case_memory or CaseMemory()

    def record_feedback(
        self,
        ticket_id: str,
        ticket_reopened: bool = False,
        analyst_override: bool = False,
        user_rating: int = 4,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Persist feedback, compute reward, update case memory.
        Returns the computed reward score.
        """
        signals = FeedbackSignals(
            ticket_reopened=ticket_reopened,
            analyst_override=analyst_override,
            user_rating=user_rating,
        )
        reward = self.reward_calc.compute(signals)

        self.conn.execute(
            """
            INSERT INTO feedback
                (ticket_id, ticket_reopened, analyst_override, user_rating, reward_score, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                int(ticket_reopened),
                int(analyst_override),
                user_rating,
                reward,
                datetime.utcnow().isoformat(),
                json.dumps(extra_meta or {}),
            ),
        )
        self.conn.commit()

        # Propagate reward into case memory
        self.case_memory.update_reward(ticket_id, reward)

        return reward

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent feedback records for the UI."""
        cursor = self.conn.execute(
            "SELECT ticket_id, ticket_reopened, analyst_override, user_rating, reward_score, created_at "
            "FROM feedback ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "ticket_id": r[0],
                "ticket_reopened": bool(r[1]),
                "analyst_override": bool(r[2]),
                "user_rating": r[3],
                "reward_score": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def get_avg_reward(self) -> float:
        """Return the global average reward score."""
        cursor = self.conn.execute("SELECT AVG(reward_score) FROM feedback")
        row = cursor.fetchone()
        return round(row[0] or 0.0, 3)

    # ── Evaluation Metrics ────────────────────────────────────────────────
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics that prove the system improves over time.
        Returns a dict with key performance indicators.
        """
        cursor = self.conn.execute(
            "SELECT ticket_id, ticket_reopened, analyst_override, user_rating, "
            "reward_score, created_at FROM feedback ORDER BY id ASC"
        )
        rows = cursor.fetchall()

        if not rows:
            return {
                "total_interactions": 0,
                "auto_resolve_rate": 0.0,
                "escalation_rate": 0.0,
                "avg_reward": 0.0,
                "avg_user_rating": 0.0,
                "reopen_rate": 0.0,
                "override_rate": 0.0,
                "first_half_reward": 0.0,
                "second_half_reward": 0.0,
                "improvement": 0.0,
            }

        total = len(rows)
        reopens = sum(1 for r in rows if r[1])
        overrides = sum(1 for r in rows if r[2])
        avg_rating = sum(r[3] for r in rows) / total
        avg_reward = sum(r[4] for r in rows) / total

        # Split into halves to show improvement trend
        mid = max(1, total // 2)
        first_half = rows[:mid]
        second_half = rows[mid:]

        first_reward = sum(r[4] for r in first_half) / len(first_half) if first_half else 0
        second_reward = sum(r[4] for r in second_half) / len(second_half) if second_half else 0
        improvement = second_reward - first_reward

        first_reopen = sum(1 for r in first_half if r[1]) / len(first_half) * 100 if first_half else 0
        second_reopen = sum(1 for r in second_half if r[1]) / len(second_half) * 100 if second_half else 0

        first_override = sum(1 for r in first_half if r[2]) / len(first_half) * 100 if first_half else 0
        second_override = sum(1 for r in second_half if r[2]) / len(second_half) * 100 if second_half else 0

        # Rolling reward (last 5 vs first 5)
        first_5_reward = sum(r[4] for r in rows[:min(5, total)]) / min(5, total)
        last_5_reward = sum(r[4] for r in rows[-min(5, total):]) / min(5, total)

        return {
            "total_interactions": total,
            "avg_reward": round(avg_reward, 3),
            "avg_user_rating": round(avg_rating, 1),
            "reopen_rate": round(reopens / total * 100, 1),
            "override_rate": round(overrides / total * 100, 1),
            "first_half_reward": round(first_reward, 3),
            "second_half_reward": round(second_reward, 3),
            "improvement": round(improvement, 3),
            "first_half_reopen_pct": round(first_reopen, 1),
            "second_half_reopen_pct": round(second_reopen, 1),
            "first_half_override_pct": round(first_override, 1),
            "second_half_override_pct": round(second_override, 1),
            "first_5_avg_reward": round(first_5_reward, 3),
            "last_5_avg_reward": round(last_5_reward, 3),
        }
