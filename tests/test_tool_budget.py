from __future__ import annotations

from src.orchestrator.tool_budget import ToolBudget


def test_tool_budget_rejects_actions_after_limit() -> None:
    budget = ToolBudget(2)

    assert budget.reserve()
    assert budget.reserve()
    assert not budget.reserve()
    assert budget.snapshot() == {
        "limit": 2,
        "used": 2,
        "remaining": 0,
        "rejected": 1,
    }

