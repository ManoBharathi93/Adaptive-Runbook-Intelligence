"""
Quick test to verify your LLM endpoint is reachable and responding.

Usage:
    python test_llm.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import get_config

def test_connection():
    cfg = get_config()
    llm_cfg = cfg.llm
    provider = llm_cfg.provider.lower()

    print("=" * 50)
    print("  LLM Connection Test")
    print("=" * 50)
    print(f"  Provider  : {provider}")
    print(f"  Model     : {llm_cfg.model_name}")

    if provider == "custom":
        print(f"  Base URL  : {llm_cfg.custom_base_url}")
        print(f"  API Key   : {'*' * 8}...{llm_cfg.custom_api_key[-4:] if len(llm_cfg.custom_api_key) > 4 else '(not set)'}")
    elif provider == "ollama":
        print(f"  Base URL  : {llm_cfg.ollama_base_url}")
    elif provider == "openai":
        k = llm_cfg.openai_api_key
        print(f"  API Key   : {'*' * 8}...{k[-4:] if len(k) > 4 else '(not set)'}")
    print("=" * 50)
    print()

    # --- Step 1: Raw HTTP test ---
    print("[1/2] Testing raw HTTP connection...")
    try:
        import httpx
        if provider == "custom":
            url = llm_cfg.custom_base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {llm_cfg.custom_api_key}", "Content-Type": "application/json"}
        elif provider == "ollama":
            url = llm_cfg.ollama_base_url.rstrip("/") + "/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
        elif provider == "openai":
            url = (llm_cfg.openai_base_url.rstrip("/") if llm_cfg.openai_base_url else "https://api.openai.com/v1") + "/chat/completions"
            headers = {"Authorization": f"Bearer {llm_cfg.openai_api_key}", "Content-Type": "application/json"}
        else:
            print("  Skipping raw HTTP test for Azure (uses SDK).")
            url = None

        if url:
            payload = {
                "model": llm_cfg.model_name,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 20,
            }
            resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                data = resp.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"  ✅ HTTP OK (status {resp.status_code})")
                print(f"  Response: \"{reply.strip()}\"")
            else:
                print(f"  ❌ HTTP {resp.status_code}")
                print(f"  Body: {resp.text[:300]}")
                return False
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False

    # --- Step 2: LangChain integration test ---
    print()
    print("[2/2] Testing LangChain ChatModel integration...")
    try:
        from graph.agents import _build_llm
        from langchain_core.messages import HumanMessage
        llm = _build_llm()
        response = llm.invoke([HumanMessage(content="Reply with only the word OK.")])
        print(f"  ✅ LangChain OK")
        print(f"  Response: \"{response.content.strip()}\"")
    except Exception as e:
        print(f"  ❌ LangChain call failed: {e}")
        return False

    print()
    print("=" * 50)
    print("  ✅ All tests passed! Your LLM is working.")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
