"""
MCP Actions – simulated HR / IT operations.

Each function represents an action the system can execute.
In production these would call real APIs; here they return
structured results for the PoC.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Any


def reset_password(user_id: str = "user@company.com") -> Dict[str, Any]:
    """Simulate an Active Directory password reset."""
    temp_password = f"Tmp{uuid.uuid4().hex[:8]}!"
    return {
        "action": "reset_password",
        "status": "success",
        "user_id": user_id,
        "temp_password": temp_password,
        "note": "Temporary password issued. Must be changed on first login within 24h.",
        "timestamp": datetime.utcnow().isoformat(),
    }


def unlock_account(user_id: str = "user@company.com") -> Dict[str, Any]:
    """Simulate unlocking a locked AD account."""
    return {
        "action": "unlock_account",
        "status": "success",
        "user_id": user_id,
        "note": "Account unlock completed. Failed‑login counter reset to 0.",
        "timestamp": datetime.utcnow().isoformat(),
    }


def grant_vpn_access(
    user_id: str = "user@company.com",
    duration_days: int = 90,
) -> Dict[str, Any]:
    """Simulate granting VPN access."""
    return {
        "action": "grant_vpn_access",
        "status": "success",
        "user_id": user_id,
        "expires": f"{duration_days} days from today",
        "vpn_profile": "corp-split-tunnel",
        "note": f"VPN token provisioned for {duration_days} days.",
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_hr_ticket(
    subject: str = "HR Request",
    description: str = "",
    priority: str = "Medium",
) -> Dict[str, Any]:
    """Simulate creating an HR service‑desk ticket."""
    ticket_id = f"HR-{uuid.uuid4().hex[:6].upper()}"
    return {
        "action": "create_hr_ticket",
        "status": "success",
        "ticket_id": ticket_id,
        "subject": subject,
        "priority": priority,
        "note": f"HR ticket {ticket_id} created and assigned to HR Operations queue.",
        "timestamp": datetime.utcnow().isoformat(),
    }


def escalate_to_analyst(
    reason: str = "",
    severity: str = "High",
) -> Dict[str, Any]:
    """Simulate escalation to a human analyst."""
    return {
        "action": "escalate_to_analyst",
        "status": "escalated",
        "assigned_to": "Tier-2 Analyst Queue",
        "reason": reason,
        "severity": severity,
        "note": "Ticket escalated. Expected response within SLA.",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Action registry ──────────────────────────────────────────────────────────
ACTION_REGISTRY: Dict[str, callable] = {
    "reset_password": reset_password,
    "unlock_account": unlock_account,
    "grant_vpn_access": grant_vpn_access,
    "create_hr_ticket": create_hr_ticket,
    "escalate_to_analyst": escalate_to_analyst,
}


def execute_action(action_name: str, **kwargs) -> Dict[str, Any]:
    """Look up and execute an action by name."""
    fn = ACTION_REGISTRY.get(action_name)
    if fn is None:
        return {
            "action": action_name,
            "status": "error",
            "note": f"Unknown action: {action_name}",
        }
    try:
        return fn(**kwargs)
    except Exception as e:
        return {
            "action": action_name,
            "status": "error",
            "note": str(e),
        }
