"""
What a model call costs, in credits, for the chat's footer (2026-09-26,
gap-closure plan item 2.4).

Two sources, and the footer says which it used:

* Robutler's models REPORT the cost: the platform's `response.done` usage
  carries `total_cost`, the credits its settle deducted for the call (B4,
  2026-09-28; the platform side ships with the final-billing lane), or the
  older `cost` (`{input_cost, output_cost, total_cost, currency:
  'credits'}`); the proxy skill passes either on in the usage chunk, and that
  number is shown as it is, the charge rather than an estimate. `total_cost`
  wins when both are there.
* A provider key has no bill to read, so the cost is ESTIMATED from the table
  below: the provider's list price per 1M tokens, in credits (1 credit = 1
  USD, the settled rate), input tokens times the input price plus output
  tokens times the output price. Cache reads and long-context tiers are not
  counted, which is why an estimate is written with a tilde. A model the
  table does not know shows tokens alone; so does a local model
  (`ollama/...`), which costs nothing.

THE TABLE ROTS. The rows are the provider list prices as the platform's own
catalog records them on 2026-09-26; before a release, check each against the
provider's list. The TypeScript SDK keeps the same table (`llm/pricing.ts`),
and both must equal the shared fixture `tests/fixtures/w2ops/cost.json`, so a
price cannot change in one SDK only.

Credits copy (CLAUDE.md): credits are named as credits; the number is the
cost of the person's own use, never a payment to anyone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, Mapping, Optional, Tuple

#: `provider/model` -> (credits per 1M input tokens, credits per 1M output tokens).
PROVIDER_LIST_PRICES: Dict[str, Tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.6),
    "openai/gpt-4.1": (2.0, 8.0),
    "openai/o3": (2.0, 8.0),
    "openai/o4-mini": (1.1, 4.4),
    "openai/gpt-5.4-nano": (0.2, 1.25),
    "openai/gpt-5.4-mini": (0.75, 4.5),
    "openai/gpt-5.4": (2.5, 15.0),
    "openai/gpt-5.5": (5.0, 30.0),
    "openai/gpt-5.6-luna": (0.2, 1.2),
    "openai/gpt-5.6-terra": (2.0, 12.0),
    "openai/gpt-5.6-sol": (4.0, 20.0),
    "openai/gpt-6-astra": (10.0, 50.0),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
    "anthropic/claude-sonnet-4-6": (3.0, 15.0),
    "anthropic/claude-opus-4-6": (5.0, 25.0),
    "anthropic/claude-opus-4-7": (5.0, 25.0),
    "anthropic/claude-opus-4-8": (5.0, 25.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "anthropic/claude-opus-5": (5.0, 25.0),
    "google/gemini-2.5-flash": (0.3, 2.5),
    "google/gemini-2.5-pro": (1.25, 10.0),
    "google/gemini-3-flash": (0.5, 3.0),
    "google/gemini-3.1-flash-lite": (0.25, 1.5),
    "google/gemini-3.1-pro": (2.0, 12.0),
    "google/gemini-3.5-flash-lite": (0.3, 2.5),
    "google/gemini-3.5-flash": (1.5, 9.0),
    "google/gemini-3.6-flash": (1.5, 7.5),
    "google/gemini-3.7-flash": (1.5, 7.5),
    "google/gemini-3.8-flash": (1.5, 7.5),
    "xai/grok-3": (3.0, 15.0),
    "xai/grok-3-mini": (0.3, 0.5),
    "xai/grok-4-0709": (3.0, 15.0),
    "xai/grok-4-fast-reasoning": (0.2, 0.5),
    "xai/grok-4-fast-non-reasoning": (0.2, 0.5),
    "xai/grok-code-fast-1": (0.2, 1.5),
    "xai/grok-4.3": (1.25, 2.5),
    "xai/grok-4.20-reasoning": (1.25, 2.5),
    "xai/grok-4.20-non-reasoning": (1.25, 2.5),
    "fireworks/deepseek-v3p2": (0.56, 1.68),
    "fireworks/deepseek-v3p1": (0.56, 1.68),
    "fireworks/deepseek-r1": (0.56, 1.68),
    "fireworks/kimi-k3": (3.0, 15.0),
    "fireworks/kimi-k2p6": (0.95, 4.0),
    "fireworks/glm-5": (1.0, 3.2),
    "fireworks/qwen3-8b": (0.2, 0.2),
    "fireworks/gpt-oss-120b": (0.15, 0.6),
    "fireworks/gpt-oss-20b": (0.07, 0.3),
    "fireworks/llama-v3p3-70b-instruct": (0.9, 0.9),
    "fireworks/minimax-m2p5": (0.3, 1.2),
}


def price_row_for(model: Optional[str]) -> Optional[str]:
    """The table row a model id finds: exact, else without a trailing
    `-YYYYMMDD` date (`claude-haiku-4-5-20251001`), else without a `:tag`. A
    bare id with no provider finds nothing: the provider is part of the price."""
    if not model or "/" not in model:
        return None
    if model in PROVIDER_LIST_PRICES:
        return model
    undated = re.sub(r"-\d{8}$", "", model)
    if undated in PROVIDER_LIST_PRICES:
        return undated
    untagged = re.sub(r":[^/]*$", "", undated)
    if untagged in PROVIDER_LIST_PRICES:
        return untagged
    return None


def estimate_cost_credits(model: Optional[str], input_tokens: int, output_tokens: int) -> Optional[float]:
    """The estimate for a call, in credits, or None for a model the table does not know."""
    row = price_row_for(model)
    if row is None:
        return None
    input_price, output_price = PROVIDER_LIST_PRICES[row]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def format_credits(credits: float) -> str:
    """A number of credits as the footer writes it: up to four decimals,
    trailing zeros dropped, thousands grouped, and `<0.0001` for a positive
    amount too small to show."""
    if not credits > 0:
        return "0"
    if credits < 0.0001:
        return "<0.0001"
    # Half up, as the TypeScript footer rounds: `round()` is half-to-even
    # (0.00425 -> 0.0042 there, 0.0043 here), and the two must agree.
    rounded = Decimal(repr(float(credits))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    whole, fraction = f"{rounded:.4f}".split(".")
    grouped = f"{int(whole):,}"
    trimmed = fraction.rstrip("0")
    return f"{grouped}.{trimmed}" if trimmed else grouped


def cost_words(credits: float, estimated: bool) -> str:
    """The footer's words: `0.0042 credits` as reported, `~0.0042 credits` as estimated."""
    return f"{'~' if estimated else ''}{format_credits(credits)} credits"


def _credits_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and number >= 0 and number != float("inf") else None


def reported_cost_credits(usage: Optional[Mapping[str, Any]]) -> Optional[float]:
    """The credits a platform-reported usage carries, when it carries any:
    `total_cost` (the charge, B4), else `cost.total_cost` in credits. Pinned
    with the TypeScript `reportedCostCredits` by
    `tests/fixtures/w2ops/final_sdk_total_cost.json`."""
    if not isinstance(usage, Mapping):
        return None
    charged = _credits_number(usage.get("total_cost"))
    if charged is not None:
        return charged
    cost = usage.get("cost")
    if not isinstance(cost, Mapping):
        return None
    total = _credits_number(cost.get("total_cost"))
    if total is None:
        return None
    currency = cost.get("currency")
    if isinstance(currency, str) and currency.lower() != "credits":
        return None
    return total


@dataclass(frozen=True)
class RunningCost:
    """A conversation's running cost: what was reported or estimated, whether
    any part is an estimate (the tilde), and whether anything was added at
    all (nothing shows tokens alone)."""

    credits: float = 0.0
    estimated: bool = False
    known: bool = False


NO_COST = RunningCost()


def add_turn_cost(
    running: RunningCost,
    model: Optional[str],
    input_tokens: int,
    output_tokens: int,
    reported: Optional[float] = None,
) -> RunningCost:
    """One turn's usage added to the running cost: the platform's number when
    it reported one, else the estimate for `model`, else nothing."""
    if reported is not None:
        return RunningCost(running.credits + reported, running.estimated, True)
    estimate = estimate_cost_credits(model, input_tokens, output_tokens)
    if estimate is None:
        return running
    return RunningCost(running.credits + estimate, True, True)
