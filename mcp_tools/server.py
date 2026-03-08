"""
FastMCP Server – exposes HR / IT actions as MCP tools.

Run standalone:  python -m mcp.server
Or import `mcp_server` to use programmatically.
"""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_tools.actions import (
    reset_password,
    unlock_account,
    grant_vpn_access,
    create_hr_ticket,
    escalate_to_analyst,
)

mcp_server = FastMCP("Adaptive-RAG-MCP")


# ── Register tools ──────────────────────────────────────────────────────────
@mcp_server.tool()
def tool_reset_password(user_id: str = "user@company.com") -> dict:
    """Reset a user's Active Directory password."""
    return reset_password(user_id)


@mcp_server.tool()
def tool_unlock_account(user_id: str = "user@company.com") -> dict:
    """Unlock a locked Active Directory account."""
    return unlock_account(user_id)


@mcp_server.tool()
def tool_grant_vpn_access(user_id: str = "user@company.com", duration_days: int = 90) -> dict:
    """Grant VPN access to a user for a specified duration."""
    return grant_vpn_access(user_id, duration_days)


@mcp_server.tool()
def tool_create_hr_ticket(subject: str = "HR Request", description: str = "", priority: str = "Medium") -> dict:
    """Create an HR service desk ticket."""
    return create_hr_ticket(subject, description, priority)


@mcp_server.tool()
def tool_escalate_to_analyst(reason: str = "", severity: str = "High") -> dict:
    """Escalate the current ticket to a human analyst."""
    return escalate_to_analyst(reason, severity)


# ── Direct execution (stdio transport) ──────────────────────────────────────
if __name__ == "__main__":
    mcp_server.run()
