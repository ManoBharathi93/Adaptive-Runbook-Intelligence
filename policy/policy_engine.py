"""
Score-based Policy Engine.

Computes a weighted decision score from:
    - LLM confidence
    - Similar-case success rate
    - Reopen probability (from case memory)
    - Action risk level
    - Runbook confidence (when available)

Decision buckets:
    AUTO_RESOLVE  |  PARTIAL_AUTOMATION  |  ESCALATE_TO_ANALYST

NO hard-coded rules — every outcome is derived from the score.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from utils.config import get_config

_pcfg = get_config().policy


class PolicyDecision(str, Enum):
    AUTO_RESOLVE = "AUTO_RESOLVE"
    PARTIAL_AUTOMATION = "PARTIAL_AUTOMATION"
    ESCALATE_TO_ANALYST = "ESCALATE_TO_ANALYST"


class PolicyResult(BaseModel):
    decision: PolicyDecision
    score: float = Field(description="Weighted composite score in [0, 1]")
    breakdown: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class PolicyEngine:
    """Stateless policy evaluator."""

    def __init__(self):
        self.cfg = _pcfg

    def evaluate(
        self,
        llm_confidence: float,
        risk_level: str,
        similar_cases: List[Dict[str, Any]],
        runbook_confidence: Optional[float] = None,
    ) -> PolicyResult:
        """
        Parameters
        ----------
        llm_confidence     : float – model's self-reported confidence (0-1)
        risk_level         : str   – "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
        similar_cases      : list  – results from CaseMemory.search()
        runbook_confidence : float – optional runbook success rate (boosts score)
        """
        case_success = self._case_success_rate(similar_cases)
        reopen_prob = self._reopen_probability(similar_cases)
        risk_num = self.cfg.risk_map.get(risk_level.upper(), 0.5)

        # If a runbook confidence is provided, blend it in
        effective_confidence = llm_confidence
        if runbook_confidence is not None:
            effective_confidence = max(llm_confidence, runbook_confidence)

        score = (
            self.cfg.w_confidence * effective_confidence
            + self.cfg.w_case_success * case_success
            - self.cfg.w_reopen * reopen_prob
            - self.cfg.w_risk * risk_num
        )
        score = max(0.0, min(1.0, score))

        if score >= self.cfg.auto_resolve_threshold:
            decision = PolicyDecision.AUTO_RESOLVE
        elif score >= self.cfg.partial_auto_threshold:
            decision = PolicyDecision.PARTIAL_AUTOMATION
        else:
            decision = PolicyDecision.ESCALATE_TO_ANALYST

        breakdown = {
            "llm_confidence": round(effective_confidence, 3),
            "case_success_rate": round(case_success, 3),
            "reopen_probability": round(reopen_prob, 3),
            "risk_numeric": round(risk_num, 3),
            "weighted_score": round(score, 3),
        }

        explanation = (
            f"Score {score:.3f} → {decision.value}. "
            f"Confidence={effective_confidence:.2f}, CaseSuccess={case_success:.2f}, "
            f"ReopenProb={reopen_prob:.2f}, Risk={risk_level}({risk_num:.1f})."
        )

        return PolicyResult(
            decision=decision, score=score,
            breakdown=breakdown, explanation=explanation,
        )

    @staticmethod
    def _case_success_rate(cases: List[Dict[str, Any]]) -> float:
        if not cases:
            return 0.5
        successes = sum(1 for c in cases if c.get("reward_score", 0) > 0)
        return successes / len(cases)

    @staticmethod
    def _reopen_probability(cases: List[Dict[str, Any]]) -> float:
        if not cases:
            return 0.2
        total_weight = 0.0
        weighted_reopen = 0.0
        for c in cases:
            sim = c.get("similarity", 0.5)
            reopen = int(c.get("reopen_count", 0))
            weighted_reopen += sim * min(reopen, 3)
            total_weight += sim
        if total_weight == 0:
            return 0.2
        raw = weighted_reopen / total_weight
        return min(raw / 3.0, 1.0)
