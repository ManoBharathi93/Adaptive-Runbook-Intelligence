"""
LangGraph Workflow – orchestrates the multi-agent pipeline.

Two EXPLICIT execution paths:
    EXPLORATORY  – full multi-agent reasoning (cold start / no matching runbook)
    FAST_PATH    – runbook similarity lookup → policy gate → direct MCP execution

Two execution modes:
    STATELESS       – docs RAG only, no case memory, no runbook intelligence
    RUNBOOK_AWARE   – docs + case memory + runbook intelligence (fast/exploratory)

Graph nodes (Exploratory):
    retrieve  → reason → policy → execute → create_runbook

Graph nodes (Fast Path):
    runbook_lookup → fast_execute
"""

from __future__ import annotations

import time
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, END

from graph.agents import (
    RetrieverAgent, ReasoningAgent, PolicyAgent, ExecutorAgent,
    RunbookLookupAgent, RunbookExecutor, TokenCounter, compute_determinism_hash,
)
from memory.docs_store import DocsStore
from memory.case_memory import CaseMemory, CaseRecord
from memory.runbook_store import RunbookStore, Runbook, RunbookStatus


# ═══════════════════════════════════════════════════════════════════════════════
# Mode types
# ═══════════════════════════════════════════════════════════════════════════════

ExecutionMode = Literal["STATELESS", "RUNBOOK_AWARE"]
ExecutionPath = Literal["EXPLORATORY", "FAST_PATH"]


# ═══════════════════════════════════════════════════════════════════════════════
# Workflow State
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowState(TypedDict, total=False):
    # Input
    query: str
    ticket_id: str
    mode: str            # "STATELESS" or "RUNBOOK_AWARE"

    # Retriever output
    docs: List[Dict[str, Any]]
    cases: List[Dict[str, Any]]

    # Reasoning output
    proposal: Dict[str, Any]

    # Policy output
    policy_result: Dict[str, Any]

    # Executor output
    execution_result: Dict[str, Any]

    # Runbook fast-path
    matched_runbook: Optional[Dict[str, Any]]
    execution_path: str   # "EXPLORATORY" or "FAST_PATH"

    # Meta
    error: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience runner (the main entry point)
# ═══════════════════════════════════════════════════════════════════════════════

def run_workflow(
    query: str,
    docs_store: Optional[DocsStore] = None,
    case_memory: Optional[CaseMemory] = None,
    runbook_store: Optional[RunbookStore] = None,
    ticket_id: Optional[str] = None,
    mode: ExecutionMode = "RUNBOOK_AWARE",
) -> Dict[str, Any]:
    """
    Execute the full pipeline with explicit path forking.

    STATELESS mode:       always EXPLORATORY, docs-only RAG, no runbooks
    RUNBOOK_AWARE mode:   check runbook store first →
                            FAST_PATH  if KNOWN_GOOD runbook found
                            EXPLORATORY otherwise

    Returns the final state dict with _instrumentation metrics.
    """
    ticket_id = ticket_id or f"TKT-{uuid.uuid4().hex[:6].upper()}"
    ds = docs_store or DocsStore()
    cm = case_memory or CaseMemory()
    rs = runbook_store or RunbookStore()

    start_time = time.time()
    agent_nodes_executed = 0
    mcp_actions_executed = 0
    execution_path: ExecutionPath = "EXPLORATORY"
    matched_runbook_info: Optional[Dict[str, Any]] = None
    llm_metrics: Dict[str, Any] = {}
    runbook_used = False
    steps_executed: List[str] = []
    runbook_id_trace: Optional[str] = None
    runbook_status_trace: Optional[str] = None
    runbook_similarity_trace: Optional[float] = None

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Runbook Lookup (RUNBOOK_AWARE only)
    # ══════════════════════════════════════════════════════════════════════
    if mode == "RUNBOOK_AWARE":
        lookup_agent = RunbookLookupAgent(runbook_store=rs)
        match = lookup_agent.lookup(query, min_similarity=0.75)
        agent_nodes_executed += 1

        if match and match["runbook"].status == RunbookStatus.KNOWN_GOOD:
            # ── FAST PATH ─────────────────────────────────────────────
            execution_path = "FAST_PATH"
            matched_runbook_info = {
                "runbook_id": match["runbook"].runbook_id,
                "trigger_text": match["trigger_text"],
                "similarity": match["similarity"],
                "steps": match["runbook"].steps,
                "status": match["runbook"].status,
                "success_count": match["runbook"].success_count,
                "success_rate": match["runbook"].success_rate,
            }
            runbook_id_trace = match["runbook"].runbook_id
            runbook_status_trace = match["runbook"].status
            runbook_similarity_trace = match["similarity"]
            runbook_used = True
            steps_executed = match["runbook"].steps

            # Fast-path policy gate: verify runbook is safe
            rb = match["runbook"]
            fast_policy_score = (
                0.4 * rb.success_rate +
                0.3 * match["similarity"] +
                0.2 * (1.0 - rb.reopen_rate) +
                0.1 * (1.0 if rb.risk_level == "LOW" else 0.5 if rb.risk_level == "MEDIUM" else 0.1)
            )

            if fast_policy_score >= 0.55:
                # Execute runbook directly — skip LLM
                executor = RunbookExecutor()
                exec_result = executor.execute_runbook(rb)
                agent_nodes_executed += 1
                mcp_actions_executed += len(rb.steps)

                total_latency_ms = (time.time() - start_time) * 1000

                # Update runbook usage counters
                rs.record_execution(
                    runbook_id=rb.runbook_id,
                    success=exec_result["status"] == "success",
                    latency_ms=total_latency_ms,
                    tokens=0,
                )

                decision = "AUTO_RESOLVE" if exec_result["status"] == "success" else "PARTIAL_AUTOMATION"
                det_hash = compute_determinism_hash(query, rb.steps, decision)

                state = {
                    "query": query,
                    "ticket_id": ticket_id,
                    "mode": mode,
                    "docs": [],
                    "cases": [],
                    "proposal": {
                        "proposed_action": rb.steps[0] if rb.steps else "unknown",
                        "confidence": rb.success_rate,
                        "reasoning": f"Fast-path: runbook {rb.runbook_id[:8]}… matched "
                                     f"(sim={match['similarity']:.2f}, "
                                     f"success_rate={rb.success_rate:.0%}). "
                                     f"Skipped exploratory reasoning.",
                        "risk_level": rb.risk_level,
                    },
                    "policy_result": {
                        "decision": decision,
                        "score": fast_policy_score,
                        "breakdown": {
                            "success_rate": round(rb.success_rate, 3),
                            "similarity": round(match["similarity"], 3),
                            "reopen_safety": round(1.0 - rb.reopen_rate, 3),
                            "risk_factor": round(0.1 * (1.0 if rb.risk_level == "LOW" else 0.5 if rb.risk_level == "MEDIUM" else 0.1), 3),
                        },
                        "explanation": f"Fast-path policy gate: score={fast_policy_score:.3f}",
                    },
                    "execution_result": exec_result,
                    "matched_runbook": matched_runbook_info,
                    "execution_path": "FAST_PATH",
                    "_instrumentation": {
                        "ticket_id": ticket_id,
                        "mode": mode,
                        "execution_path": "FAST_PATH",
                        "tokens_prompt": 0,
                        "tokens_completion": 0,
                        "tokens_total": 0,
                        "llm_calls": 0,
                        "latency_ms": round(total_latency_ms, 1),
                        "llm_latency_ms": 0,
                        "agents_invoked": agent_nodes_executed,
                        "mcp_actions_executed": mcp_actions_executed,
                        "policy_decision": decision,
                        "escalated": False,
                        "confidence": rb.success_rate,
                        "memory_hit": True,
                        "runbook_used": True,
                        "runbook_id": rb.runbook_id,
                        "runbook_status": rb.status,
                        "runbook_similarity": match["similarity"],
                        "cases_retrieved": 0,
                        "docs_retrieved": 0,
                        "policy_score": fast_policy_score,
                        "steps_executed": rb.steps,
                        "determinism_hash": det_hash,
                    },
                }
                return state

            # Policy gate rejected fast path — fall through to exploratory
            execution_path = "EXPLORATORY"
            matched_runbook_info["gate_rejected"] = True
            runbook_id_trace = match["runbook"].runbook_id
            runbook_status_trace = match["runbook"].status
            runbook_similarity_trace = match["similarity"]

    # ══════════════════════════════════════════════════════════════════════
    # EXPLORATORY PATH (full multi-agent reasoning)
    # ══════════════════════════════════════════════════════════════════════

    # ── Retrieve ──────────────────────────────────────────────────────────
    retriever = RetrieverAgent(docs_store=ds, case_memory=cm)
    agent_nodes_executed += 1

    if mode == "STATELESS":
        doc_hits = retriever.docs.search(query)
        case_hits = []
    else:
        result = retriever.run(query)
        doc_hits = result["docs"]
        case_hits = result["cases"]

    # ── Reason (LLM call) ─────────────────────────────────────────────────
    reasoning_agent = ReasoningAgent()
    agent_nodes_executed += 1

    if mode == "STATELESS":
        proposal = reasoning_agent.run(query=query, docs=doc_hits, cases=[])
    else:
        proposal = reasoning_agent.run(query=query, docs=doc_hits, cases=case_hits)

    llm_metrics = proposal.pop("_metrics", {})

    # ── Policy ────────────────────────────────────────────────────────────
    policy_agent = PolicyAgent()
    agent_nodes_executed += 1

    if mode == "STATELESS":
        policy_result = policy_agent.run(proposal=proposal, cases=[])
    else:
        policy_result = policy_agent.run(proposal=proposal, cases=case_hits)

    # ── Execute ───────────────────────────────────────────────────────────
    executor_agent = ExecutorAgent()
    agent_nodes_executed += 1
    exec_result = executor_agent.run(proposal=proposal, policy_result=policy_result)
    action_name = proposal.get("proposed_action", "escalate_to_analyst")
    steps_executed = [action_name]
    mcp_actions_executed += 1

    total_latency_ms = (time.time() - start_time) * 1000

    # ── Create runbook candidate (RUNBOOK_AWARE + non-escalation) ─────────
    if mode == "RUNBOOK_AWARE" and policy_result["decision"] != "ESCALATE_TO_ANALYST":
        rb = rs.create_runbook(
            trigger_text=query,
            steps=steps_executed,
            risk_level=proposal.get("risk_level", "MEDIUM"),
            latency_ms=total_latency_ms,
            tokens=llm_metrics.get("tokens_total", 0),
        )
        runbook_used = False  # created, not reused
        runbook_id_trace = rb.runbook_id
        runbook_status_trace = rb.status
        if runbook_similarity_trace is None:
            runbook_similarity_trace = 1.0
        agent_nodes_executed += 1  # runbook creation counts as a node

    # ── Persist case into memory (RUNBOOK_AWARE only) ─────────────────────
    if mode == "RUNBOOK_AWARE":
        new_case = CaseRecord(
            ticket_id=ticket_id,
            problem=query,
            query=query,
            context={"source": "workflow", "mode": mode},
            actions_taken=steps_executed,
            outcome=policy_result.get("decision", "unknown"),
            escalation_level=1 if policy_result.get("decision") == "ESCALATE_TO_ANALYST" else 0,
            reopen_count=0,
            reward_score=0.0,
        )
        cm.add_case(new_case)

    det_hash = compute_determinism_hash(
        query, steps_executed, policy_result.get("decision", "UNKNOWN")
    )

    state = {
        "query": query,
        "ticket_id": ticket_id,
        "mode": mode,
        "docs": doc_hits,
        "cases": case_hits,
        "proposal": proposal,
        "policy_result": policy_result,
        "execution_result": exec_result,
        "matched_runbook": matched_runbook_info,
        "execution_path": "EXPLORATORY",
        "_instrumentation": {
            "ticket_id": ticket_id,
            "mode": mode,
            "execution_path": "EXPLORATORY",
            "tokens_prompt": llm_metrics.get("tokens_prompt", 0),
            "tokens_completion": llm_metrics.get("tokens_completion", 0),
            "tokens_total": llm_metrics.get("tokens_total", 0),
            "llm_calls": llm_metrics.get("llm_calls", 1),
            "latency_ms": round(total_latency_ms, 1),
            "llm_latency_ms": llm_metrics.get("latency_ms", 0),
            "agents_invoked": agent_nodes_executed,
            "mcp_actions_executed": mcp_actions_executed,
            "policy_decision": policy_result.get("decision", "UNKNOWN"),
            "escalated": policy_result.get("decision") == "ESCALATE_TO_ANALYST",
            "confidence": float(proposal.get("confidence", 0)),
            "memory_hit": len(case_hits) > 0,
            "runbook_used": runbook_used,
            "runbook_id": runbook_id_trace,
            "runbook_status": runbook_status_trace,
            "runbook_similarity": runbook_similarity_trace,
            "cases_retrieved": len(case_hits),
            "docs_retrieved": len(doc_hits),
            "policy_score": policy_result.get("score", 0),
            "steps_executed": steps_executed,
            "determinism_hash": det_hash,
        },
    }
    return state
