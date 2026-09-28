"""
`webagents budget <tokenId>`: the budget tree of a run (webagents gap-closure
plan 2.3, 2026-09-26), in the words the TypeScript CLI uses
(`typescript/src/cli/budget-tree.ts`), so the two print the same lines; the
fixture `tests/fixtures/payments/delegate_budget.json` pins them.

The tree comes from `GET /api/payments/tokens/{id}/tree` on the platform: the
root token and every child a hop derived from it, each with its budget, what
was charged against it and what it has left. Credits, never a currency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config_store import cli_command


def format_credits(value: Any) -> str:
    """A credit amount as the tree prints it: at most 9 decimals, no trailing zeros."""
    rounded = round(float(value or 0), 9)
    if rounded == int(rounded):
        return str(int(rounded))
    return repr(rounded)


def render_budget_tree(root: Dict[str, Any]) -> List[str]:
    """One line per token, two spaces per level; `audience=any` for an unrestricted token."""
    lines: List[str] = []

    def walk(node: Dict[str, Any], depth: int) -> None:
        audience = node.get("audience") or []
        lines.append(
            f"{'  ' * depth}{node['id']} audience={','.join(audience) if audience else 'any'} "
            f"budget={format_credits(node.get('budgetCredits'))} spent={format_credits(node.get('spentCredits'))} "
            f"remaining={format_credits(node.get('remainingCredits'))} {node.get('status') or 'unknown'}"
        )
        for child in node.get("children") or []:
            walk(child, depth + 1)

    walk(root, 0)
    return lines


def budget_tree_totals(root: Dict[str, Any]) -> str:
    """The totals line under the tree: budgets and spend summed over every token."""
    budget = 0.0
    spent = 0.0
    count = 0

    def walk(node: Dict[str, Any]) -> None:
        nonlocal budget, spent, count
        budget += float(node.get("budgetCredits") or 0)
        spent += float(node.get("spentCredits") or 0)
        count += 1
        for child in node.get("children") or []:
            walk(child)

    walk(root)
    return f"total budget={format_credits(budget)} spent={format_credits(spent)} across {count} token{'' if count == 1 else 's'}"


@dataclass
class BudgetTree:
    ok: bool
    message: str = ""
    code: str = ""
    fix: str = ""
    tree: Optional[Dict[str, Any]] = None
    lines: List[str] = field(default_factory=list)
    totals: str = ""


def budget_tree(token_id: str) -> BudgetTree:
    """Fetch and render the tree for `token_id` with the CLI's sign-in."""
    import httpx

    from .config_store import platform_url
    from .credentials import get_token

    portal = platform_url().rstrip("/")
    host = re.sub(r"^https?://", "", portal)
    token = get_token()
    if not token:
        return BudgetTree(False, f"Not signed in to {host}.", "not_signed_in", f"Run `{cli_command('login')}`.")
    if not re.match(r"^[0-9a-f-]{36}$", token_id or "", re.IGNORECASE):
        return BudgetTree(False, f"{token_id} is not a payment token id.", "bad_token_id", "Pass a payment token id from your token list on the platform; a delegate receipt names only a fingerprint.")
    try:
        response = httpx.get(f"{portal}/api/payments/tokens/{token_id}/tree", headers={"Authorization": f"Bearer {token}"}, timeout=8)
    except httpx.HTTPError as error:
        return BudgetTree(False, f"Could not reach {host}: {error}", "unreachable", "Check the network, or `webagents config get platform.url`.")
    if response.status_code == 401:
        return BudgetTree(False, f"Your sign-in on {host} has expired.", "expired", f"Run `{cli_command('login')}`.")
    if response.status_code == 404:
        return BudgetTree(False, f"No budget tree for {token_id} on {host}: not a token of yours.", "not_found", "")
    if response.status_code >= 400:
        return BudgetTree(False, f"{host} answered {response.status_code}.", "http_error", "")
    try:
        tree = response.json().get("tree")
    except Exception:
        tree = None
    if not isinstance(tree, dict) or not isinstance(tree.get("id"), str):
        return BudgetTree(False, f"{host} answered without a tree.", "bad_answer", "")
    return BudgetTree(True, tree=tree, lines=render_budget_tree(tree), totals=budget_tree_totals(tree))
