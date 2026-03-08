# 📘 Adaptive Runbook Intelligence Platform

> A production-grade proof-of-concept that demonstrates how an agentic system can **learn from its own execution history**, build a curated runbook library, and **skip the LLM entirely** for known-good automations — with measurable, reproducible evidence.

---


## Why This Exists

Modern LLM-based automation has a blind spot: **every request is treated as a cold start.** Even if the system has successfully resolved the same category of issue 100 times, it still:
- Calls the LLM to reason from scratch
- Burns tokens re-deriving the same conclusion
- Introduces non-determinism from model sampling
- Requires the same latency every time

**This PoC proves a better approach is possible.**

---

## How It Works — Two Execution Paths

```
                          ┌──────────────────────────┐
                          │    Incoming Query         │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │  Runbook Similarity Search │
                          │  (ChromaDB embeddings)     │
                          └────────────┬─────────────┘
                                       │
                           ┌───────────┴───────────┐
                           │                       │
                    Match found?              No match
                    KNOWN_GOOD?
                    Score ≥ 0.55?
                           │                       │
                    ┌──────▼──────┐         ┌──────▼──────┐
                    │  FAST PATH  │         │ EXPLORATORY │
                    │             │         │             │
                    │ 0 LLM calls │         │ Retrieve    │
                    │ 0 tokens    │         │ Reason      │
                    │ Direct MCP  │         │ Policy      │
                    │ execution   │         │ Execute     │
                    │ <50ms       │         │ Create      │
                    │             │         │ Runbook     │
                    └─────────────┘         └─────────────┘
```

| Path | When it fires | What happens | LLM involved? |
|------|--------------|--------------|---------------|
| **FAST_PATH** | KNOWN_GOOD runbook matched with high confidence | Runbook steps → policy gate → direct MCP execution | **No** |
| **EXPLORATORY** | Cold start / no matching runbook / low confidence | Full agent pipeline: retrieve → reason → policy → execute | **Yes** |

---

## Why This Is Not Fine-Tuning

| | Fine-Tuning | Runbook Intelligence |
|-|------------|---------------------|
| **What changes** | Model weights | System routing |
| **Risk** | Model degradation, catastrophic forgetting | Zero model risk — model is untouched |
| **Reversibility** | Requires retraining | Delete the runbook |
| **Audit trail** | Opaque weight changes | Named runbooks with success/failure counters |
| **Determinism** | Still probabilistic | FAST_PATH is fully deterministic |
| **LLM dependency** | High (still calls the model) | Zero for FAST_PATH |
| **Time to learn** | Hours/days of training | Immediate — first successful execution |

**The core insight:** We are not making the model smarter. We are making the **system** smarter.

---




## Architecture

```
adaptive-agentic-rag/
├── app.py                       # Streamlit UI
├── run_benchmark.py             # 3-phase benchmark runner
├── graph/
│   ├── agents.py                # 6 agent classes (incl. RunbookLookup, RunbookExecutor)
│   └── workflow.py              # Two-path orchestration (FAST_PATH / EXPLORATORY)
├── memory/
│   ├── docs_store.py            # ChromaDB knowledge base (KB docs)
│   ├── case_memory.py           # ChromaDB episodic memory (past resolutions)
│   └── runbook_store.py         # ChromaDB + SQLite runbook library
├── policy/
│   ├── policy_engine.py         # Score-based policy gate (runbook-aware)
│   └── reward_calculator.py     # Feedback → reward signal
├── feedback/
│   └── feedback_collector.py    # SQLite feedback persistence
├── mcp_tools/
│   └── actions.py               # MCP action registry (5 simulated actions)
├── metrics/
│   └── comparator.py            # Before/after comparison engine
├── utils/
│   ├── config.py                # Multi-provider LLM configuration
│   └── embeddings.py            # sentence-transformers wrapper
├── data/
│   ├── synthetic_tickets.json   # 20 test tickets with ground truth
│   ├── hr_docs.txt              # HR knowledge base
│   └── it_docs.txt              # IT knowledge base
├── demo/
│   └── DEMO_SCRIPT.md           # 5-act presenter walkthrough
└── README.md                    # This file
```

### Key Components

**Runbook Store** (`memory/runbook_store.py`)
- Dual-backend: ChromaDB for embedding similarity search + SQLite for structured counters
- Runbook schema: `runbook_id`, `trigger_text`, `trigger_embedding`, `steps`, `success_count`, `failure_count`, `reopen_count`, `avg_latency_ms`, `avg_tokens`, `risk_level`, `status`, `created_at`, `last_used_at`
- Status promotion: EXPERIMENTAL → KNOWN_GOOD (≥3 successes, ≥80% success rate)
- Auto-demotion: KNOWN_BAD (≥3 failures OR >50% reopen rate)
- KNOWN_BAD runbooks are automatically excluded from similarity search

**Workflow** (`graph/workflow.py`)
- Explicit path forking: check runbook store → policy gate → route to FAST_PATH or EXPLORATORY
- FAST_PATH: 0 LLM calls, 0 tokens, direct MCP execution of runbook steps
- EXPLORATORY: full agent pipeline with token/call instrumentation
- Every execution produces a determinism hash: `sha256(query + steps + outcome)`

**Policy Engine** (`policy/policy_engine.py`)
- Score-based with configurable thresholds (≥0.70 auto-resolve, ≥0.45 partial, else escalate)
- Accepts optional `runbook_confidence` parameter for fast-path scoring
- Weighted formula: 0.4×success_rate + 0.3×similarity + 0.2×(1−reopen_rate) + 0.1×risk_factor

---

## Runbook Lifecycle

```
  ┌──────────────┐     3+ successes     ┌──────────────┐
  │ EXPERIMENTAL │  ─────────────────►  │  KNOWN_GOOD  │
  │              │     ≥80% success     │              │
  └──────┬───────┘                      └──────────────┘
         │
         │  3+ failures
         │  OR >50% reopen
         ▼
  ┌──────────────┐
  │  KNOWN_BAD   │  ← excluded from matching
  └──────────────┘
```

- Every execution updates counters (`success_count`, `failure_count`, `reopen_count`)
- `avg_latency_ms` and `avg_tokens` are recomputed as running averages
- Status is recomputed after every update — no manual promotion needed

---

## Quick Start

### Prerequisites

- Python 3.10+
- An LLM provider (OpenAI, Azure, Ollama, or custom Vertex/Athena endpoint)

### Install

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Configure

```powershell
# Option 1: Custom endpoint (Vertex/Athena)
$env:LLM_PROVIDER = "custom"
$env:CUSTOM_LLM_BASE_URL = "https://your-endpoint.net/vertex"
$env:CUSTOM_LLM_API_KEY = "your-api-key"
$env:LLM_MODEL = "meta/llama-4-maverick-17b-128e-instruct-maas"

# Option 2: OpenAI
$env:LLM_PROVIDER = "openai"
$env:OPENAI_API_KEY = "sk-..."

# Option 3: Ollama (local)
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:LLM_MODEL = "llama3"
```

### Run Benchmark

```bash
python run_benchmark.py
```

This runs 20 tickets through 3 phases:

| Phase | Mode | What happens |
|-------|------|-------------|
| 1 — Stateless | STATELESS | Docs RAG only, always EXPLORATORY |
| 2 — Runbook Build | RUNBOOK_AWARE | Case memory + creates EXPERIMENTAL runbooks |
| 3 — Runbook Reuse | RUNBOOK_AWARE | KNOWN_GOOD runbooks → FAST_PATH for matching tickets |

Plus a determinism test (same query 3×, verify identical hashes).

Output: `metrics/comparison_summary.json`

### Launch UI

```bash
streamlit run app.py
```

---

## What the Benchmark Proves

The benchmark runs the **same 20 tickets** through all three phases and produces a comparison report showing:

| Metric | What it measures |
|--------|-----------------|
| `token_reduction_pct` | How many fewer tokens Phase 3 uses vs Phase 1 |
| `latency_reduction_pct` | How much faster Phase 3 is vs Phase 1 |
| `escalation_reduction_pct` | How many fewer tickets get escalated |
| `fast_path_ratio` | What percentage of Phase 3 tickets hit FAST_PATH |
| `llm_calls_reduction_pct` | How many fewer LLM calls Phase 3 makes |
| `agent_steps_skipped_avg` | How many agent steps are bypassed on average |
| `determinism_improvement` | Whether FAST_PATH produces consistent outcomes |

**Every number comes from real execution.** No mocks, no stubs, no pre-computed results.

---

## Why This Is Safe for Production

1. **The LLM model is never modified.** Same weights, same API, same prompts across all phases.
2. **FAST_PATH is fully deterministic.** Same query → same runbook → same steps → same outcome. Auditable via determinism hash.
3. **Automatic quality control.** Bad runbooks are demoted to KNOWN_BAD and excluded. The system self-heals.
4. **Graceful fallback.** If no matching runbook exists, or if the runbook starts failing, the system falls back to full exploratory reasoning. No degradation.
5. **Policy gate always runs.** Even FAST_PATH goes through the policy engine before MCP execution. High-risk actions require higher confidence.
6. **Full audit trail.** Every runbook has named steps, success/failure counters, timestamps, and a status lifecycle.

---

## How It Plugs Into Existing Systems

| Component | Current (PoC) | Production Replacement |
|-----------|---------------|----------------------|
| Knowledge Base | `.txt` files indexed in ChromaDB | Confluence / SharePoint / S3 document stores |
| Case Memory | ChromaDB | Enterprise vector DB (Pinecone, Weaviate, pgvector) |
| Runbook Store | ChromaDB + SQLite | PostgreSQL + vector extension |
| MCP Actions | 5 simulated actions | Real ServiceNow / Jira / AD API calls |
| LLM | Custom Vertex endpoint | Any OpenAI-compatible API |
| Feedback | SQLite | ITSM feedback loop (ServiceNow incident lifecycle) |
| UI | Streamlit | Embedded in existing ITSM portal |

The architecture is **provider-agnostic** and **integration-ready**. The agent pipeline, runbook store, and policy engine are all swappable backends.

---

## Key Design Decisions

**Q: Why ChromaDB + SQLite instead of just ChromaDB?**
> ChromaDB handles embedding similarity search efficiently. SQLite handles structured counters (success_count, failure_count) and complex queries (status aggregation, threshold checks). Both are file-based with zero infrastructure requirements for a PoC.

**Q: Why not use LangGraph for the two-path routing?**
> Explicitness. The FAST_PATH vs EXPLORATORY fork is a fundamental architectural decision, not a graph routing concern. Making it an explicit `if/else` in the orchestrator makes the code auditable and the instrumentation straightforward.

**Q: Why hash(query + steps + outcome) for determinism?**
> It proves that the same input produces the same output without storing full execution traces. Governance teams can compare hashes across runs to verify consistency.

**Q: Why 3 successes for promotion and not 1?**
> One success could be a fluke. Three successes with ≥80% success rate gives statistical confidence. The threshold is configurable.

---

### Results:
<img width="1534" height="453" alt="image" src="https://github.com/user-attachments/assets/c55cadd1-02a8-40de-8d90-5c4dfbb47c8a" />

<img width="1466" height="837" alt="image" src="https://github.com/user-attachments/assets/bfdb5985-8d6f-4cb3-ad83-aeb7a37e186a" />

### Link to article:
https://medium.com/@immanobharathi21/why-should-ai-re-solve-the-same-problem-every-time-6d102f114043


