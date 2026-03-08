"""
Streamlit UI — Runbook Intelligence Platform (Demo-Grade)

Features:
    - Enter query manually
    - Toggle mode: STATELESS vs RUNBOOK_AWARE
    - Show: retrieved docs, matched runbook, execution path, actions, tokens, latency
    - Benchmark dashboard with before/after charts
    - Runbook library viewer
    - Determinism proof section

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from memory.docs_store import DocsStore
from memory.case_memory import CaseMemory
from memory.runbook_store import RunbookStore
from graph.workflow import run_workflow
from feedback.feedback_collector import FeedbackCollector
from utils.config import get_config, DB_DIR

# ═══════════════════════════════════════════════════════════════════════════════
# Labels
# ═══════════════════════════════════════════════════════════════════════════════

_ACTION_LABELS = {
    "reset_password": "🔑 Reset Password",
    "unlock_account": "🔓 Unlock Account",
    "grant_vpn_access": "🌐 Grant VPN Access",
    "create_hr_ticket": "📋 Create HR Ticket",
    "escalate_to_analyst": "👤 Escalate to Analyst",
    "runbook_execution": "📘 Runbook Execution",
}

_DECISION_LABELS = {
    "AUTO_RESOLVE": ("✅", "Auto-Resolved", "green"),
    "PARTIAL_AUTOMATION": ("⚙️", "Partially Automated", "orange"),
    "ESCALATE_TO_ANALYST": ("👤", "Escalated to Analyst", "red"),
}

_PATH_LABELS = {
    "FAST_PATH": ("⚡", "Fast Path (Runbook)", "green"),
    "EXPLORATORY": ("🔍", "Exploratory (Full Reasoning)", "blue"),
}

_RISK_EMOJI = {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High"}
_SATISFACTION_EMOJI = {1: "😡", 2: "😟", 3: "😐", 4: "🙂", 5: "😍"}

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Runbook Intelligence Platform",
    page_icon="📘",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Init stores
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_stores():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    ds = DocsStore()
    ds.index_documents()
    cm = CaseMemory()
    cm.seed_from_json()
    rs = RunbookStore()
    fc = FeedbackCollector(case_memory=cm)
    return ds, cm, rs, fc

docs_store, case_memory, runbook_store, feedback_collector = init_stores()

# ═══════════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════════

_defaults = {
    "history": [],
    "execution_mode": "RUNBOOK_AWARE",
    "llm_provider": os.environ.get("LLM_PROVIDER", "custom"),
    "llm_model": os.environ.get("LLM_MODEL", "meta/llama-4-maverick-17b-128e-instruct-maas"),
    "custom_base_url": os.environ.get("CUSTOM_LLM_BASE_URL", "https://apisix-dp.athena-preprod.otxlab.net/vertex"),
    "custom_api_key": os.environ.get("CUSTOM_LLM_API_KEY", ""),
    "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "similarity_threshold": 0.35,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _apply_sidebar_config():
    cfg = get_config()
    cfg.llm.provider = st.session_state.llm_provider
    cfg.llm.model_name = st.session_state.llm_model
    cfg.llm.custom_base_url = st.session_state.custom_base_url
    cfg.llm.custom_api_key = st.session_state.custom_api_key
    cfg.llm.ollama_base_url = st.session_state.ollama_base_url
    cfg.llm.openai_api_key = st.session_state.openai_api_key
    cfg.retrieval.similarity_threshold = st.session_state.similarity_threshold

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuration")

    # Mode toggle
    st.subheader("Execution Mode")
    mode = st.radio(
        "Mode",
        ["STATELESS", "RUNBOOK_AWARE"],
        index=1 if st.session_state.execution_mode == "RUNBOOK_AWARE" else 0,
        key="execution_mode",
        help="STATELESS = Docs RAG only. RUNBOOK_AWARE = Docs + Case Memory + Runbook Intelligence.",
    )

    st.divider()

    # LLM Provider
    provider_options = ["openai", "azure", "ollama", "custom"]
    st.selectbox("LLM Provider", provider_options,
                 index=provider_options.index(st.session_state.llm_provider),
                 key="llm_provider")

    if st.session_state.llm_provider == "custom":
        st.text_input("Custom Base URL", key="custom_base_url")
        st.text_input("API Key", key="custom_api_key", type="password")
        st.text_input("Model", key="llm_model")
    elif st.session_state.llm_provider == "ollama":
        st.text_input("Ollama URL", key="ollama_base_url")
        st.text_input("Model", key="llm_model")
    elif st.session_state.llm_provider == "openai":
        st.text_input("OpenAI Key", key="openai_api_key", type="password")
        st.text_input("Model", key="llm_model")

    st.caption(f"Active: **{st.session_state.llm_provider}** / `{st.session_state.llm_model}`")

    st.divider()
    st.slider("Similarity threshold", 0.0, 1.0, step=0.05, key="similarity_threshold")

    if st.button("🔌 Test LLM Connection"):
        _apply_sidebar_config()
        try:
            from graph.agents import _build_llm
            llm = _build_llm()
            with st.spinner("Pinging LLM…"):
                resp = llm.invoke("Reply with OK")
            st.success(f"Connected: {resp.content[:80]}")
        except Exception as exc:
            st.error(f"Failed: {exc}")

    st.divider()
    st.header("📊 Quick Stats")
    c1, c2 = st.columns(2)
    c1.metric("Cases", case_memory.collection.count())
    c2.metric("KB Chunks", docs_store.collection.count())
    c3, c4 = st.columns(2)
    c3.metric("Runbooks", runbook_store.total_count())
    c4.metric("Interactions", len(st.session_state.history))


# ═══════════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════════

st.title("📘 Runbook Intelligence Platform")
st.markdown(
    "A production-grade system that **learns which automation sequences actually work**, "
    "stores known-good runbooks, and **skips exploratory reasoning when confidence is high** — "
    "using real execution data, not heuristics."
)

# Mode indicator
mode_emoji = "🔍" if st.session_state.execution_mode == "STATELESS" else "📘"
mode_label = "Stateless (Docs RAG Only)" if st.session_state.execution_mode == "STATELESS" else "Runbook-Aware (Full Intelligence)"
st.info(f"{mode_emoji} **Mode: {mode_label}**")

# Query input
query = st.text_area(
    "Describe your HR / IT issue:",
    placeholder="e.g. I cannot access VPN after my password was reset",
    height=80,
)

col_submit, col_clear = st.columns([1, 4])
with col_submit:
    submitted = st.button("🚀 Submit", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# Run workflow & display
# ═══════════════════════════════════════════════════════════════════════════════

if (submitted and query.strip()) or st.session_state.history:
    result = None

    if submitted and query.strip():
        _apply_sidebar_config()
        with st.spinner("Processing through the agent pipeline…"):
            try:
                result = run_workflow(
                    query=query.strip(),
                    docs_store=docs_store,
                    case_memory=case_memory,
                    runbook_store=runbook_store,
                    mode=st.session_state.execution_mode,
                )
                st.session_state.history.append(result)
            except Exception as e:
                st.error(f"Error: {e}")
                result = None
    else:
        result = st.session_state.history[-1]

    if result:
        proposal = result.get("proposal", {})
        policy = result.get("policy_result", {})
        execution = result.get("execution_result", {})
        instrumentation = result.get("_instrumentation", {})
        ticket_id = result.get("ticket_id", "?")
        exec_path = instrumentation.get("execution_path", "EXPLORATORY")

        # ── Execution path banner ─────────────────────────────────────
        path_emoji, path_label, path_color = _PATH_LABELS.get(exec_path, ("❓", exec_path, "gray"))
        st.markdown(f"### {path_emoji} Execution Path: :{path_color}[{path_label}]")

        # ── Decision banner ───────────────────────────────────────────
        decision_raw = policy.get("decision", "N/A")
        dec_emoji, dec_label, dec_color = _DECISION_LABELS.get(decision_raw, ("❓", decision_raw, "gray"))
        st.markdown(f"### {dec_emoji} Outcome: :{dec_color}[{dec_label}]")

        # ── Metrics row ───────────────────────────────────────────────
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Tokens", instrumentation.get("tokens_total", 0))
        mc2.metric("Latency", f"{instrumentation.get('latency_ms', 0):.0f}ms")
        mc3.metric("LLM Calls", instrumentation.get("llm_calls", 0))
        mc4.metric("Agents", instrumentation.get("agents_invoked", 0))
        mc5.metric("MCP Actions", instrumentation.get("mcp_actions_executed", 0))
        mc6.metric("Confidence", f"{instrumentation.get('confidence', 0):.0%}")

        # ── Result cards ──────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Recommended Action**")
            action_raw = proposal.get("proposed_action", "N/A")
            st.markdown(f"#### {_ACTION_LABELS.get(action_raw, action_raw)}")
        with c2:
            st.markdown("**Policy Score**")
            st.markdown(f"#### {policy.get('score', 0):.3f}")
        with c3:
            risk_raw = proposal.get("risk_level", "unknown").lower()
            st.markdown("**Risk Level**")
            st.markdown(f"#### {_RISK_EMOJI.get(risk_raw, risk_raw)}")

        # ── Reasoning ─────────────────────────────────────────────────
        st.info(f"💡 **Reasoning:** {proposal.get('reasoning', 'N/A')}")

        # ── Execution result ──────────────────────────────────────────
        exec_note = execution.get("note", "") or execution.get("message", "")
        if execution.get("status") == "success":
            st.success(f"⚡ **Executed.** {exec_note}")
        elif execution.get("status") in ("escalated", "skipped"):
            st.warning(f"⏭️ **Escalated.** {exec_note}")
            st.info(
                "In RUNBOOK_AWARE mode, escalated outcomes do not create reusable runbooks. "
                "FAST_PATH appears only after successful automations are repeated and promoted to KNOWN_GOOD."
            )
        else:
            st.info(f"Result: {exec_note}")

        # ── Matched runbook (if any) ──────────────────────────────────
        matched_rb = result.get("matched_runbook")
        if matched_rb:
            with st.expander("📘 Matched Runbook"):
                st.json(matched_rb)

        # ── Retrieved docs ────────────────────────────────────────────
        with st.expander("📚 Retrieved Documents"):
            docs = result.get("docs", [])
            if docs:
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**{i}. {d['source']}** (relevance: {d['similarity']:.0%})")
                    st.caption(d["text"][:300])
            else:
                st.write("No docs retrieved (fast-path skips doc retrieval)." if exec_path == "FAST_PATH" else "No matching docs.")

        # ── Retrieved cases ───────────────────────────────────────────
        with st.expander("🗃️ Similar Past Cases"):
            cases = result.get("cases", [])
            if cases:
                for i, c in enumerate(cases, 1):
                    reward = c.get("reward_score", 0)
                    emoji = "✅" if reward > 0 else "⚠️" if reward == 0 else "❌"
                    st.markdown(f"**{i}. {c['ticket_id']}** {emoji} — sim {c['similarity']:.0%}, reward {reward:+.2f}")
            else:
                st.write("No similar cases." if st.session_state.execution_mode == "STATELESS" else "No cases (fast-path or first run).")

        # ── Policy breakdown ──────────────────────────────────────────
        with st.expander("📐 Policy Score Breakdown"):
            breakdown = policy.get("breakdown", {})
            if breakdown:
                for k, v in breakdown.items():
                    st.write(f"**{k.replace('_', ' ').title()}:** {v:.3f}")
            st.caption(policy.get("explanation", ""))

        # ── Full instrumentation ──────────────────────────────────────
        with st.expander("🔧 Full Instrumentation Record"):
            st.json(instrumentation)

        # ── Feedback ──────────────────────────────────────────────────
        st.divider()
        st.subheader("💬 Feedback")
        fb1, fb2, fb3 = st.columns(3)
        with fb1:
            reopened = st.checkbox("❗ Reopened", key=f"reopen_{ticket_id}")
        with fb2:
            overridden = st.checkbox("🔄 Analyst override", key=f"override_{ticket_id}")
        with fb3:
            rating = st.select_slider(
                "Satisfaction", options=[1, 2, 3, 4, 5], value=4,
                format_func=lambda x: f"{_SATISFACTION_EMOJI[x]} {x}/5",
                key=f"rating_{ticket_id}",
            )

        if st.button("📨 Submit Feedback", key=f"fb_{ticket_id}"):
            reward = feedback_collector.record_feedback(
                ticket_id=ticket_id, ticket_reopened=reopened,
                analyst_override=overridden, user_rating=rating,
            )
            # Also update runbook if one was used
            rb_id = instrumentation.get("runbook_id")
            if rb_id:
                runbook_store.record_execution(
                    runbook_id=rb_id,
                    success=not reopened and not overridden,
                    reopened=reopened,
                    latency_ms=instrumentation.get("latency_ms", 0),
                    tokens=instrumentation.get("tokens_total", 0),
                )
            st.success(f"Feedback recorded — reward: **{reward:+.3f}**")


# ═══════════════════════════════════════════════════════════════════════════════
# 📘 Runbook Library
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📘 Runbook Library")

all_runbooks = runbook_store.list_all()
if all_runbooks:
    status_counts = runbook_store.count_by_status()
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Runbooks", len(all_runbooks))
    sc2.metric("🟢 KNOWN_GOOD", status_counts.get("KNOWN_GOOD", 0))
    sc3.metric("🟡 EXPERIMENTAL", status_counts.get("EXPERIMENTAL", 0))
    sc4.metric("🔴 KNOWN_BAD", status_counts.get("KNOWN_BAD", 0))

    with st.expander("View All Runbooks"):
        for rb in all_runbooks:
            status_emoji = "🟢" if rb.status == "KNOWN_GOOD" else "🟡" if rb.status == "EXPERIMENTAL" else "🔴"
            st.markdown(
                f"**{status_emoji} {rb.runbook_id[:8]}…** — "
                f"Steps: `{' → '.join(rb.steps)}` | "
                f"Success: {rb.success_count}/{rb.total_uses} ({rb.success_rate:.0%}) | "
                f"Status: {rb.status}"
            )
            st.caption(f"Trigger: {rb.trigger_text[:80]}…")
else:
    st.info("No runbooks yet. Submit queries in RUNBOOK_AWARE mode to build the library.")


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 Benchmark Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📊 Benchmark Dashboard — Before vs After")
st.caption(
    "Run `python run_benchmark.py` to execute 20 tickets in 3 phases: "
    "Stateless → Runbook Build → Runbook Reuse. "
    "The gap between Phase 1 and Phase 3 proves measurable improvement."
)

METRICS_DIR = PROJECT_ROOT / "metrics"
COMPARISON_FILE = METRICS_DIR / "comparison_summary.json"

if COMPARISON_FILE.exists():
    with open(COMPARISON_FILE, encoding="utf-8") as f:
        full_report = json.load(f)

    def _to_float_pct(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace("%", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    # Pull out comparison sections
    cmp_reuse = full_report.get("stateless_vs_runbook_reuse", {})
    cmp_build = full_report.get("stateless_vs_runbook_build", {})
    phases = full_report.get("phases", {})
    det_test = full_report.get("determinism_test", {})

    p1 = phases.get("phase1_stateless", {})
    p2 = phases.get("phase2_runbook_build", {})
    p3 = phases.get("phase3_runbook_reuse", {})

    # ── Executive summary (non-technical view) ───────────────────────
    if p1 and p3:
        token_reduction = _to_float_pct(cmp_reuse.get("token_reduction_pct"))
        latency_reduction = _to_float_pct(cmp_reuse.get("latency_reduction_pct"))
        escalation_reduction = _to_float_pct(cmp_reuse.get("escalation_reduction_pct"))
        llm_reduction = _to_float_pct(cmp_reuse.get("llm_calls_reduction_pct"))

        fast_path_count = p3.get("fast_path_count", 0)
        total_tickets = p3.get("total", 0)
        fast_path_ratio = (fast_path_count / total_tickets * 100) if total_tickets else 0

        st.markdown("#### 📌 Executive Summary (What improved)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "Tokens",
            f"{p1.get('avg_tokens', 0):.0f} → {p3.get('avg_tokens', 0):.0f}",
            f"{abs(token_reduction):.1f}% lower" if token_reduction is not None else "N/A",
        )
        s2.metric(
            "Latency",
            f"{p1.get('avg_latency_ms', 0):.0f}ms → {p3.get('avg_latency_ms', 0):.0f}ms",
            f"{abs(latency_reduction):.1f}% faster" if latency_reduction is not None else "N/A",
        )
        s3.metric(
            "Escalations",
            f"{p1.get('escalated', 0)} → {p3.get('escalated', 0)}",
            f"{abs(escalation_reduction):.1f}% fewer" if escalation_reduction is not None else "N/A",
        )
        s4.metric(
            "LLM Calls",
            f"{p1.get('avg_llm_calls', 0):.2f} → {p3.get('avg_llm_calls', 0):.2f}",
            f"{abs(llm_reduction):.1f}% fewer" if llm_reduction is not None else "N/A",
        )

        st.info(
            f"Out of {total_tickets} tickets, {fast_path_count} hit FAST_PATH "
            f"({fast_path_ratio:.0f}%). Those tickets skip the LLM and execute known-good runbooks directly."
        )

        if det_test.get("is_deterministic"):
            st.success("Determinism check passed: repeated runs produced the same outcome hash.")
        else:
            st.warning("Determinism check is mixed: exploratory path can vary across runs.")

    # ── Phase summary cards ───────────────────────────────────────────
    if p1 and p2 and p3:
        st.markdown("#### Phase Summary")
        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            st.markdown("##### 🔵 Phase 1: Stateless")
            st.caption("Docs RAG only — always EXPLORATORY")
            st.metric("Escalated", p1.get("escalated", 0))
            st.metric("Auto-Resolve", p1.get("auto_resolve", 0))
            st.metric("Avg Tokens", f"{p1.get('avg_tokens', 0):.0f}")
            st.metric("Avg Latency", f"{p1.get('avg_latency_ms', 0):.0f} ms")
            st.metric("Avg Agents", f"{p1.get('avg_agents', 0):.1f}")
            st.metric("LLM Calls", p1.get("total_llm_calls", 0))

        with pc2:
            st.markdown("##### 🟡 Phase 2: Runbook Build")
            st.caption("Case memory + builds runbook library")
            st.metric("Escalated", p2.get("escalated", 0))
            st.metric("Auto-Resolve", p2.get("auto_resolve", 0))
            st.metric("Fast Path", p2.get("fast_path_count", 0))
            st.metric("Avg Tokens", f"{p2.get('avg_tokens', 0):.0f}")
            st.metric("Avg Latency", f"{p2.get('avg_latency_ms', 0):.0f} ms")
            st.metric("Runbooks Created", p2.get("runbook_reuse_count", 0))

        with pc3:
            st.markdown("##### 🟢 Phase 3: Runbook Reuse")
            st.caption("KNOWN_GOOD runbooks → FAST_PATH")
            st.metric("Escalated", p3.get("escalated", 0))
            st.metric("Auto-Resolve", p3.get("auto_resolve", 0))
            st.metric("⚡ Fast Path", p3.get("fast_path_count", 0))
            st.metric("Avg Tokens", f"{p3.get('avg_tokens', 0):.0f}")
            st.metric("Avg Latency", f"{p3.get('avg_latency_ms', 0):.0f} ms")
            st.metric("Runbooks Reused", p3.get("runbook_reuse_count", 0))

    # ── Headline metrics (Stateless → Runbook Reuse) ──────────────────
    st.markdown("---")
    st.markdown("#### Key Improvements: Stateless → Runbook Reuse (Phase 1 → Phase 3)")

    h1, h2, h3 = st.columns(3)
    h1.metric("🪙 Token Δ", cmp_reuse.get("token_reduction_pct", "N/A"))
    h2.metric("⏱️ Latency Δ", cmp_reuse.get("latency_reduction_pct", "N/A"))
    h3.metric("🚨 Escalation Δ", cmp_reuse.get("escalation_reduction_pct", "N/A"))

    h4, h5, h6 = st.columns(3)
    h4.metric("🤖 Agent Steps Skipped", cmp_reuse.get("agent_steps_skipped_avg", "N/A"))
    h5.metric("📘 Runbook Reuse", cmp_reuse.get("runbook_reuse_ratio", "N/A"))
    h6.metric("🧠 LLM Calls Δ", cmp_reuse.get("llm_calls_reduction_pct", "N/A"))

    # ── Determinism proof ─────────────────────────────────────────────
    if det_test:
        st.markdown("---")
        st.markdown("#### 🎯 Determinism Proof")
        det_col1, det_col2 = st.columns(2)
        with det_col1:
            st.write(f"**Query:** {det_test.get('query', '')[:80]}…")
            st.write(f"**Runs:** {det_test.get('runs', 0)}")
            st.write(f"**Unique outcomes:** {det_test.get('unique_hashes', 0)}")
        with det_col2:
            if det_test.get("is_deterministic"):
                st.success("✅ DETERMINISTIC — Same runbook, same steps, same outcome across all runs.")
            else:
                st.warning("⚠️ Non-deterministic — outcomes varied (expected for EXPLORATORY path).")
            st.caption(f"Hashes: {det_test.get('hashes', [])}")

    # ── Per-ticket table ──────────────────────────────────────────────
    per_ticket = cmp_reuse.get("per_ticket_comparison", [])
    with st.expander("📋 Per-Ticket Comparison Table"):
        if per_ticket:
            import pandas as pd
            df = pd.DataFrame(per_ticket)
            display_cols = [c for c in [
                "ticket_id", "stateless_decision", "runbook_decision",
                "stateless_path", "runbook_path",
                "stateless_tokens", "runbook_tokens",
                "stateless_latency_ms", "runbook_latency_ms",
                "stateless_agents", "runbook_agents",
                "stateless_llm_calls", "runbook_llm_calls",
                "runbook_used", "decision_changed",
            ] if c in df.columns]
            st.dataframe(df[display_cols], width="stretch", hide_index=True)
        else:
            st.write("No per-ticket data.")

    # ── Charts ────────────────────────────────────────────────────────
    if per_ticket:
        import pandas as pd
        df = pd.DataFrame(per_ticket)

        st.markdown("#### Visual Comparison")
        t1, t2, t3, t4 = st.tabs(["Tokens", "Latency", "Agents", "Execution Path"])

        with t1:
            chart = df[["ticket_id", "stateless_tokens", "runbook_tokens"]].set_index("ticket_id")
            st.bar_chart(chart)

        with t2:
            chart = df[["ticket_id", "stateless_latency_ms", "runbook_latency_ms"]].set_index("ticket_id")
            st.bar_chart(chart)

        with t3:
            chart = df[["ticket_id", "stateless_agents", "runbook_agents"]].set_index("ticket_id")
            st.bar_chart(chart)

        with t4:
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown("*Stateless Decisions*")
                st.bar_chart(df["stateless_decision"].value_counts())
            with dc2:
                st.markdown("*Runbook-Aware Decisions*")
                st.bar_chart(df["runbook_decision"].value_counts())

    # ── Detailed breakdown ────────────────────────────────────────────
    with st.expander("📐 Detailed Numeric Breakdown"):
        bl_data = cmp_reuse.get("stateless", {})
        ra_data = cmp_reuse.get("runbook_aware", {})
        if bl_data and ra_data:
            bcol, rcol = st.columns(2)
            with bcol:
                st.markdown("**🔵 Stateless**")
                for k, v in bl_data.items():
                    st.write(f"{k}: {v}")
            with rcol:
                st.markdown("**🟢 Runbook-Aware (Reuse)**")
                for k, v in ra_data.items():
                    st.write(f"{k}: {v}")

    # ── How this proves real improvement ──────────────────────────────
    with st.expander("🧪 How does this prove real improvement?"):
        st.markdown("""
**Two Explicit Execution Paths:**

| Path | When it fires | What happens | LLM involved? |
|------|--------------|--------------|---------------|
| **EXPLORATORY** | Cold start / no matching runbook | Full agent reasoning → tool discovery → policy → MCP | ✅ Yes |
| **FAST_PATH** | KNOWN_GOOD runbook matched | Runbook lookup → policy gate → direct MCP execution | ❌ No |

**What changes between phases:**

1. **Stateless (Phase 1):** Every ticket triggers full EXPLORATORY reasoning. No case memory,
   no runbooks. High latency, high token usage, frequent escalation.

2. **Runbook Build (Phase 2):** System now has case memory + creates EXPERIMENTAL runbooks
   from successful executions. Similar tickets benefit from case success rates.

3. **Runbook Reuse (Phase 3):** After feedback, runbooks get promoted to KNOWN_GOOD.
   Matching tickets now hit FAST_PATH — zero LLM calls, zero exploratory reasoning,
   direct MCP execution of the known-good automation sequence.

**This is NOT fine-tuning.** The LLM model is unchanged. The system learns by:
- Storing which automation sequences worked (runbooks)
- Tracking success/failure/reopen counters
- Promoting proven sequences and blacklisting bad ones
- Skipping the LLM entirely when the system already knows the answer

**"We are not making the model smarter. We are making the system smarter."**
""")

else:
    st.info(
        "No benchmark results found. Run the benchmark first:\n\n"
        "```powershell\n"
        "$env:CUSTOM_LLM_API_KEY='your_key'\n"
        "python run_benchmark.py\n"
        "```"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Past Interactions
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.history:
    st.divider()
    st.subheader("🕘 Past Interactions")
    for i, h in enumerate(reversed(st.session_state.history)):
        p = h.get("proposal", {})
        pol = h.get("policy_result", {})
        inst = h.get("_instrumentation", {})
        path = inst.get("execution_path", "EXPLORATORY")
        path_emoji = "⚡" if path == "FAST_PATH" else "🔍"
        dec_raw = pol.get("decision", "N/A")
        dec_emoji, dec_label, _ = _DECISION_LABELS.get(dec_raw, ("❓", dec_raw, "gray"))
        action_label = _ACTION_LABELS.get(p.get("proposed_action", ""), p.get("proposed_action", ""))

        idx = len(st.session_state.history) - i
        with st.expander(f"#{idx} {path_emoji} {action_label} → {dec_emoji} {dec_label} (score {pol.get('score', 0):.3f})"):
            st.markdown(f"**Query:** {h.get('query', '')}")
            st.markdown(f"**Path:** {path} | **Tokens:** {inst.get('tokens_total', 0)} | "
                        f"**Latency:** {inst.get('latency_ms', 0):.0f}ms | "
                        f"**LLM Calls:** {inst.get('llm_calls', 0)}")
            st.markdown(f"**Reasoning:** {p.get('reasoning', 'N/A')}")
