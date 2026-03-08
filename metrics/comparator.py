"""
Metrics Comparator – compares STATELESS vs RUNBOOK_AWARE benchmark results.

Produces the required comparison summary with:
    token_reduction_pct
    latency_reduction_pct
    agent_steps_skipped_avg
    escalation_reduction_pct
    determinism_improvement
    runbook_reuse_ratio
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

METRICS_DIR = Path(__file__).resolve().parent


def _safe_avg(values: list, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _pct_change(baseline: float, new: float) -> float:
    if baseline == 0:
        return 0.0
    return round(((new - baseline) / baseline) * 100, 2)


def compare_metrics(
    baseline: List[Dict[str, Any]],
    runbook_aware: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare STATELESS baseline vs RUNBOOK_AWARE results."""
    bl = [r for r in baseline if r.get("policy_decision") != "ERROR"]
    ra = [r for r in runbook_aware if r.get("policy_decision") != "ERROR"]

    # ── Tokens ─────────────────────────────────────────────────────────────
    avg_bl_tokens = _safe_avg([r["tokens_total"] for r in bl])
    avg_ra_tokens = _safe_avg([r["tokens_total"] for r in ra])

    # ── Latency ────────────────────────────────────────────────────────────
    avg_bl_latency = _safe_avg([r["latency_ms"] for r in bl])
    avg_ra_latency = _safe_avg([r["latency_ms"] for r in ra])

    # ── Agent steps ────────────────────────────────────────────────────────
    avg_bl_agents = _safe_avg([r.get("agents_invoked", 4) for r in bl])
    avg_ra_agents = _safe_avg([r.get("agents_invoked", 4) for r in ra])
    agent_steps_skipped = max(0, avg_bl_agents - avg_ra_agents)

    # ── Escalation ─────────────────────────────────────────────────────────
    bl_esc = sum(1 for r in bl if r.get("escalated")) / len(bl) * 100 if bl else 0
    ra_esc = sum(1 for r in ra if r.get("escalated")) / len(ra) * 100 if ra else 0

    # ── LLM calls ─────────────────────────────────────────────────────────
    avg_bl_llm = _safe_avg([r.get("llm_calls", 1) for r in bl])
    avg_ra_llm = _safe_avg([r.get("llm_calls", 0) for r in ra])

    # ── Determinism ────────────────────────────────────────────────────────
    bl_decisions = {r["ticket_id"]: r["policy_decision"] for r in bl}
    ra_decisions = {r["ticket_id"]: r["policy_decision"] for r in ra}
    common = set(bl_decisions) & set(ra_decisions)
    diffs = sum(1 for tid in common if bl_decisions[tid] != ra_decisions[tid])
    determinism_pct = round(diffs / len(common) * 100, 2) if common else 0

    # ── Determinism hash consistency (runbook-aware only) ─────────────────
    ra_hashes = [r.get("determinism_hash", "") for r in ra if r.get("runbook_used")]
    # For fast-path tickets, same query should produce same hash
    hash_unique_ratio = len(set(ra_hashes)) / len(ra_hashes) if ra_hashes else 0

    # ── Runbook metrics ────────────────────────────────────────────────────
    ra_runbook_used = sum(1 for r in ra if r.get("runbook_used"))
    runbook_reuse_ratio = round(ra_runbook_used / len(ra) * 100, 2) if ra else 0

    ra_fast_path = sum(1 for r in ra if r.get("execution_path") == "FAST_PATH")
    fast_path_ratio = round(ra_fast_path / len(ra) * 100, 2) if ra else 0

    # ── MCP actions ────────────────────────────────────────────────────────
    avg_bl_mcp = _safe_avg([r.get("mcp_actions_executed", 1) for r in bl])
    avg_ra_mcp = _safe_avg([r.get("mcp_actions_executed", 1) for r in ra])

    # ── Confidence ─────────────────────────────────────────────────────────
    avg_bl_conf = _safe_avg([r.get("confidence", 0) for r in bl])
    avg_ra_conf = _safe_avg([r.get("confidence", 0) for r in ra])

    # ── Policy score ───────────────────────────────────────────────────────
    avg_bl_score = _safe_avg([r.get("policy_score", 0) for r in bl])
    avg_ra_score = _safe_avg([r.get("policy_score", 0) for r in ra])

    # ── Auto-resolve rate ──────────────────────────────────────────────────
    bl_auto = sum(1 for r in bl if r.get("policy_decision") == "AUTO_RESOLVE")
    ra_auto = sum(1 for r in ra if r.get("policy_decision") == "AUTO_RESOLVE")

    # ── Per-ticket comparison ──────────────────────────────────────────────
    bl_by_id = {r["ticket_id"]: r for r in bl}
    ra_by_id = {r["ticket_id"]: r for r in ra}
    per_ticket = []
    for tid in sorted(common):
        b = bl_by_id[tid]
        a = ra_by_id[tid]
        per_ticket.append({
            "ticket_id": tid,
            "stateless_decision": b["policy_decision"],
            "runbook_decision": a["policy_decision"],
            "stateless_path": b.get("execution_path", "EXPLORATORY"),
            "runbook_path": a.get("execution_path", "EXPLORATORY"),
            "stateless_confidence": round(b.get("confidence", 0), 3),
            "runbook_confidence": round(a.get("confidence", 0), 3),
            "stateless_tokens": b["tokens_total"],
            "runbook_tokens": a["tokens_total"],
            "stateless_latency_ms": round(b["latency_ms"], 1),
            "runbook_latency_ms": round(a["latency_ms"], 1),
            "stateless_agents": b.get("agents_invoked", 4),
            "runbook_agents": a.get("agents_invoked", 0),
            "stateless_llm_calls": b.get("llm_calls", 1),
            "runbook_llm_calls": a.get("llm_calls", 0),
            "runbook_used": a.get("runbook_used", False),
            "runbook_id": a.get("runbook_id"),
            "decision_changed": b["policy_decision"] != a["policy_decision"],
            "determinism_hash": a.get("determinism_hash", ""),
        })

    # ── Required output format ─────────────────────────────────────────────
    summary = {
        # Mandated comparison fields
        "token_reduction_pct": f"{_pct_change(avg_bl_tokens, avg_ra_tokens)}%",
        "latency_reduction_pct": f"{_pct_change(avg_bl_latency, avg_ra_latency)}%",
        "agent_steps_skipped_avg": round(agent_steps_skipped, 2),
        "escalation_reduction_pct": f"{_pct_change(bl_esc, ra_esc)}%",
        "determinism_improvement": f"{determinism_pct}% of decisions changed with runbook context",
        "runbook_reuse_ratio": f"{runbook_reuse_ratio}%",

        # Extended metrics
        "fast_path_ratio": f"{fast_path_ratio}%",
        "avg_llm_calls_stateless": round(avg_bl_llm, 2),
        "avg_llm_calls_runbook": round(avg_ra_llm, 2),
        "llm_calls_reduction_pct": f"{_pct_change(avg_bl_llm, avg_ra_llm)}%",

        "stateless": {
            "total_tickets": len(bl),
            "avg_tokens": round(avg_bl_tokens, 1),
            "avg_latency_ms": round(avg_bl_latency, 1),
            "avg_agents_invoked": round(avg_bl_agents, 1),
            "avg_llm_calls": round(avg_bl_llm, 2),
            "escalation_rate_pct": round(bl_esc, 1),
            "auto_resolve_count": bl_auto,
            "avg_confidence": round(avg_bl_conf, 3),
            "avg_policy_score": round(avg_bl_score, 3),
        },
        "runbook_aware": {
            "total_tickets": len(ra),
            "avg_tokens": round(avg_ra_tokens, 1),
            "avg_latency_ms": round(avg_ra_latency, 1),
            "avg_agents_invoked": round(avg_ra_agents, 1),
            "avg_llm_calls": round(avg_ra_llm, 2),
            "escalation_rate_pct": round(ra_esc, 1),
            "auto_resolve_count": ra_auto,
            "avg_confidence": round(avg_ra_conf, 3),
            "avg_policy_score": round(avg_ra_score, 3),
            "fast_path_count": ra_fast_path,
            "runbook_reuse_count": ra_runbook_used,
            "runbook_reuse_ratio_pct": runbook_reuse_ratio,
        },
        "per_ticket_comparison": per_ticket,
    }
    return summary


def load_and_compare(
    stateless_path: Optional[Path] = None,
    runbook_path: Optional[Path] = None,
) -> Dict[str, Any]:
    bl_path = stateless_path or (METRICS_DIR / "stateless_metrics.json")
    ra_path = runbook_path or (METRICS_DIR / "runbook_metrics.json")

    with open(bl_path, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(ra_path, encoding="utf-8") as f:
        runbook_aware = json.load(f)

    return compare_metrics(baseline, runbook_aware)


if __name__ == "__main__":
    summary = load_and_compare()
    print(json.dumps(summary, indent=2))
