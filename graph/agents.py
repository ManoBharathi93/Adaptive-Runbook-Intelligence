"""
Agents – individual reasoning units used inside the LangGraph workflow.

Agents:
    1. RetrieverAgent      – fetches docs + past cases
    2. ReasoningAgent      – calls LLM to propose action (with token tracking)
    3. PolicyAgent         – runs policy engine, decides execution path
    4. ExecutorAgent       – calls MCP actions
    5. RunbookLookupAgent  – fast-path: queries runbook store for known-good match
    6. RunbookExecutor     – fast-path: direct MCP execution of runbook steps
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

from utils.config import get_config
from memory.docs_store import DocsStore
from memory.case_memory import CaseMemory, CaseRecord
from memory.runbook_store import RunbookStore, Runbook, RunbookStatus
from policy.policy_engine import PolicyEngine, PolicyResult
from mcp_tools.actions import execute_action


# ═══════════════════════════════════════════════════════════════════════════════
# Token-counting callback
# ═══════════════════════════════════════════════════════════════════════════════

class TokenCounter(BaseCallbackHandler):
    """LangChain callback that captures token usage from the LLM response."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs):
        self.llm_calls += 1
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    info = gen.generation_info or {}
                    usage = info.get("token_usage") or info.get("usage") or {}
                    if usage:
                        self.prompt_tokens += usage.get("prompt_tokens", 0)
                        self.completion_tokens += usage.get("completion_tokens", 0)
                        self.total_tokens += usage.get("total_tokens", 0)
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage", {})
                if usage and self.total_tokens == 0:
                    self.prompt_tokens = usage.get("prompt_tokens", 0)
                    self.completion_tokens = usage.get("completion_tokens", 0)
                    self.total_tokens = usage.get("total_tokens", 0)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# LLM factory  —  supports openai | azure | ollama | custom
# ═══════════════════════════════════════════════════════════════════════════════

def _build_llm():
    cfg = get_config().llm
    provider = cfg.provider.lower().strip()

    if provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=cfg.azure_endpoint,
            azure_deployment=cfg.azure_deployment,
            api_version=cfg.azure_api_version,
            api_key=cfg.azure_api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    if provider == "ollama":
        base = cfg.ollama_base_url.rstrip("/")
        return ChatOpenAI(
            model=cfg.model_name, base_url=f"{base}/v1",
            api_key="ollama", temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        )
    if provider == "custom":
        return ChatOpenAI(
            model=cfg.model_name,
            base_url=cfg.custom_base_url.rstrip("/"),
            api_key=cfg.custom_api_key or "no-key",
            temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        )
    # OpenAI default
    kwargs: dict = {
        "model": cfg.model_name, "api_key": cfg.openai_api_key,
        "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
    }
    if cfg.openai_base_url:
        kwargs["base_url"] = cfg.openai_base_url
    return ChatOpenAI(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Retriever Agent
# ═══════════════════════════════════════════════════════════════════════════════

class RetrieverAgent:
    """Fetches relevant docs AND similar past cases."""

    def __init__(
        self,
        docs_store: Optional[DocsStore] = None,
        case_memory: Optional[CaseMemory] = None,
    ):
        self.docs = docs_store or DocsStore()
        self.cases = case_memory or CaseMemory()

    def run(self, query: str) -> Dict[str, Any]:
        doc_hits = self.docs.search(query)
        case_hits = self.cases.search(query)
        return {"docs": doc_hits, "cases": case_hits}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Reasoning Agent (LLM)
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are an expert IT / HR support assistant.  Given a user's ticket query, \
relevant knowledge‑base excerpts, and similar past cases, propose the best \
action.

You MUST reply with ONLY a JSON object (no markdown fencing) in this schema:
{
  "proposed_action": "<one of: reset_password | unlock_account | grant_vpn_access | create_hr_ticket | escalate_to_analyst>",
  "confidence": <float 0‑1>,
  "reasoning": "<brief explanation>",
  "risk_level": "<LOW | MEDIUM | HIGH | CRITICAL>"
}

Guidelines:
- Use past case outcomes to calibrate your confidence.
- If past similar cases were often reopened or escalated, lower your confidence.
- If the request is sensitive (harassment, payroll, security), set risk to HIGH or CRITICAL.
- Only propose actions you are confident about.
"""


class ReasoningAgent:
    """Calls the LLM to produce a structured action proposal."""

    def __init__(self):
        self.llm = _build_llm()
        self.token_counter = TokenCounter()

    def run(
        self, query: str,
        docs: List[Dict[str, Any]],
        cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        doc_text = "\n\n".join(
            f"[KB – {d['source']}] (sim={d['similarity']})\n{d['text']}"
            for d in docs[:4]
        ) or "No relevant docs found."

        case_text = "\n\n".join(
            f"[Case {c['ticket_id']}] (sim={c['similarity']}, reward={c['reward_score']})\n"
            f"  Problem: {c['problem']}\n  Actions: {c['actions_taken']}\n"
            f"  Outcome: {c['outcome']} | Reopens: {c['reopen_count']}"
            for c in cases[:5]
        ) or "No similar past cases."

        user_msg = (
            f"### User Query\n{query}\n\n"
            f"### Knowledge Base Excerpts\n{doc_text}\n\n"
            f"### Similar Past Cases\n{case_text}"
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        self.token_counter = TokenCounter()
        start_time = time.time()
        response = self.llm.invoke(messages, config={"callbacks": [self.token_counter]})
        latency_ms = (time.time() - start_time) * 1000

        content = response.content.strip()

        tokens_prompt = self.token_counter.prompt_tokens
        tokens_completion = self.token_counter.completion_tokens
        tokens_total = self.token_counter.total_tokens
        llm_calls = self.token_counter.llm_calls

        if tokens_total == 0 and hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            if not usage:
                usage = response.response_metadata.get("usage", {})
            tokens_prompt = usage.get("prompt_tokens", 0)
            tokens_completion = usage.get("completion_tokens", 0)
            tokens_total = usage.get("total_tokens", tokens_prompt + tokens_completion)

        if tokens_total == 0:
            tokens_prompt = len(user_msg) // 4 + len(_SYSTEM_PROMPT) // 4
            tokens_completion = len(content) // 4
            tokens_total = tokens_prompt + tokens_completion

        # Parse JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            proposal = json.loads(content)
        except json.JSONDecodeError:
            proposal = {
                "proposed_action": "escalate_to_analyst",
                "confidence": 0.2,
                "reasoning": "Failed to parse LLM output. Escalating for safety.",
                "risk_level": "HIGH",
            }

        proposal.setdefault("proposed_action", "escalate_to_analyst")
        proposal.setdefault("confidence", 0.5)
        proposal.setdefault("reasoning", "")
        proposal.setdefault("risk_level", "MEDIUM")

        proposal["_metrics"] = {
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
            "latency_ms": round(latency_ms, 1),
            "llm_calls": llm_calls or 1,
        }

        return proposal


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Policy Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyAgent:
    def __init__(self):
        self.engine = PolicyEngine()

    def run(self, proposal: Dict[str, Any], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: PolicyResult = self.engine.evaluate(
            llm_confidence=float(proposal.get("confidence", 0.5)),
            risk_level=str(proposal.get("risk_level", "MEDIUM")),
            similar_cases=cases,
        )
        return {
            "decision": result.decision.value,
            "score": result.score,
            "breakdown": result.breakdown,
            "explanation": result.explanation,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Executor (calls MCP actions)
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutorAgent:
    """Executes the proposed action via the MCP action registry."""

    def run(self, proposal: Dict[str, Any], policy_result: Dict[str, Any]) -> Dict[str, Any]:
        decision = policy_result["decision"]

        if decision == "ESCALATE_TO_ANALYST":
            return execute_action(
                "escalate_to_analyst",
                reason=proposal.get("reasoning", "Policy engine escalated."),
                severity=proposal.get("risk_level", "High"),
            )
        action_name = proposal.get("proposed_action", "escalate_to_analyst")

        if decision == "PARTIAL_AUTOMATION":
            result = execute_action(action_name)
            result["_partial_flag"] = True
            result["note"] = (result.get("note", "") +
                              " [PARTIAL] Flagged for analyst review.")
            return result

        return execute_action(action_name)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Runbook Lookup Agent (FAST PATH)
# ═══════════════════════════════════════════════════════════════════════════════

class RunbookLookupAgent:
    """
    Fast-path agent: queries the runbook store for a matching known-good runbook.
    If found and the runbook is KNOWN_GOOD, the system can skip exploratory reasoning.
    """

    def __init__(self, runbook_store: Optional[RunbookStore] = None):
        self.store = runbook_store or RunbookStore()

    def lookup(self, query: str, min_similarity: float = 0.75) -> Optional[Dict[str, Any]]:
        return self.store.find_matching_runbook(
            query=query, min_similarity=min_similarity, exclude_bad=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Runbook Executor (FAST PATH — direct MCP execution)
# ═══════════════════════════════════════════════════════════════════════════════

class RunbookExecutor:
    """Executes a known runbook's steps directly via MCP without LLM."""

    def execute_runbook(self, runbook: Runbook) -> Dict[str, Any]:
        results = []
        all_success = True
        for step in runbook.steps:
            result = execute_action(step)
            results.append(result)
            if result.get("status") not in ("success", "escalated"):
                all_success = False

        return {
            "action": "runbook_execution",
            "runbook_id": runbook.runbook_id,
            "steps_executed": runbook.steps,
            "step_results": results,
            "status": "success" if all_success else "partial_failure",
            "note": (
                f"Runbook {runbook.runbook_id[:8]}… executed "
                f"{len(runbook.steps)} steps "
                f"({'all succeeded' if all_success else 'some failed'}). "
                f"Status: {runbook.status}"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism hashing
# ═══════════════════════════════════════════════════════════════════════════════

def compute_determinism_hash(query: str, steps: List[str], outcome: str) -> str:
    """hash(query + steps + outcome) for determinism proof."""
    content = json.dumps({
        "query": query.strip().lower(),
        "steps": sorted(steps),
        "outcome": outcome,
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
