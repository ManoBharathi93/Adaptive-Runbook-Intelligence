"""
Central configuration for the Adaptive Agentic RAG system.
All thresholds, model names, and paths are configured here.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"
SQLITE_PATH = DB_DIR / "feedback.db"
CHROMA_DOCS_DIR = str(DB_DIR / "chroma_docs")
CHROMA_CASES_DIR = str(DB_DIR / "chroma_cases")


# ── Environment helpers ────────────────────────────────────────────────────────
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ── LLM Configuration ─────────────────────────────────────────────────────────
class LLMConfig(BaseModel):
    """
    Multi-provider LLM configuration.

    Supported providers (set via LLM_PROVIDER env var):
        openai   – OpenAI API (default)
        azure    – Azure OpenAI Service
        ollama   – Local Ollama server (any open-source model)
        custom   – Any OpenAI-compatible endpoint (e.g., Vertex, vLLM, LiteLLM)
    """
    provider: str = Field(
        default_factory=lambda: _env("LLM_PROVIDER", "openai"),
        description="One of: openai | azure | ollama | custom",
    )
    model_name: str = Field(
        default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"),
        description="Model identifier. Examples: gpt-4o-mini, llama3.2, meta/llama-4-maverick-17b-128e-instruct-maas",
    )
    temperature: float = 0.1
    max_tokens: int = 2048

    # ── OpenAI ─────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = Field(
        default_factory=lambda: _env("OPENAI_BASE_URL", ""),
        description="Override base URL for OpenAI-compatible proxies",
    )

    # ── Azure OpenAI ───────────────────────────────────────────────────────
    azure_api_key: str = Field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY", ""))
    azure_endpoint: str = Field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT", ""))
    azure_deployment: str = Field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))
    azure_api_version: str = Field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", "2024-06-01"))

    # ── Ollama ─────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Ollama server URL",
    )

    # ── Custom OpenAI-compatible endpoint ──────────────────────────────────
    custom_base_url: str = Field(
        default_factory=lambda: _env(
            "CUSTOM_LLM_BASE_URL",
            "https://apisix-dp.athena-preprod.otxlab.net/vertex",
        ),
        description="Base URL for any OpenAI-compatible API",
    )
    custom_api_key: str = Field(
        default_factory=lambda: _env("CUSTOM_LLM_API_KEY", ""),
        description="Bearer token / API key for the custom endpoint",
    )


# ── Retrieval Configuration ───────────────────────────────────────────────────
class RetrievalConfig(BaseModel):
    docs_top_k: int = 4
    cases_top_k: int = 5
    similarity_threshold: float = Field(
        default=0.35,
        description="Minimum cosine similarity to include a result",
    )
    embedding_model: str = "all-MiniLM-L6-v2"


# ── Policy Engine Weights ─────────────────────────────────────────────────────
class PolicyConfig(BaseModel):
    """
    Score‑based thresholds – NO hard‑coding of outcomes.
    final_score = w1*confidence + w2*case_success - w3*reopen_prob - w4*risk
    """
    w_confidence: float = 0.35
    w_case_success: float = 0.30
    w_reopen: float = 0.20
    w_risk: float = 0.15
    auto_resolve_threshold: float = 0.70
    partial_auto_threshold: float = 0.45
    # Below partial → ESCALATE_TO_ANALYST

    risk_map: dict = {
        "LOW": 0.1,
        "MEDIUM": 0.4,
        "HIGH": 0.8,
        "CRITICAL": 1.0,
    }


# ── Reward Configuration ──────────────────────────────────────────────────────
class RewardConfig(BaseModel):
    success_reward: float = 1.0
    reopen_penalty: float = -0.5
    override_penalty: float = -0.7
    rating_weight: float = 0.2   # per-star contribution: score += (rating-3)*weight


# ── FastMCP ────────────────────────────────────────────────────────────────────
class MCPConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


# ── Global singleton ──────────────────────────────────────────────────────────
class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    policy: PolicyConfig = PolicyConfig()
    reward: RewardConfig = RewardConfig()
    mcp: MCPConfig = MCPConfig()


_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return a singleton AppConfig so all modules share the same instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance
