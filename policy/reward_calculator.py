"""
Reward Calculator – converts delayed feedback signals into a scalar reward score.

Signals:
    ticket_reopened  (bool)   → penalty
    analyst_override (bool)   → penalty
    user_rating      (1‑5)    → bonus / penalty relative to midpoint (3)

Formula:
    reward  = base_success
            + reopen_penalty   (if reopened)
            + override_penalty (if overridden)
            + (rating - 3) * rating_weight
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from utils.config import get_config

_rcfg = get_config().reward


class FeedbackSignals(BaseModel):
    ticket_reopened: bool = False
    analyst_override: bool = False
    user_rating: int = Field(default=4, ge=1, le=5)


class RewardCalculator:

    def __init__(self):
        self.cfg = _rcfg

    def compute(self, signals: FeedbackSignals) -> float:
        """Return a single float reward score."""
        score = self.cfg.success_reward  # start optimistic

        if signals.ticket_reopened:
            score += self.cfg.reopen_penalty      # negative

        if signals.analyst_override:
            score += self.cfg.override_penalty     # negative

        # Rating contribution (rating=3 is neutral)
        score += (signals.user_rating - 3) * self.cfg.rating_weight

        return round(max(-1.0, min(1.0, score)), 3)  # clamp to [-1, 1]
