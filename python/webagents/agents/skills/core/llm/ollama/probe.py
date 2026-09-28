"""
Whether Ollama answers, and which models it serves (2026-09-26, plan item
2.8): what `webagents doctor` and `webagents models` ask before they call a
local model "ready". One GET of the OpenAI-compatible model list, with a
short timeout, because a machine without Ollama must not make either command
wait. The TypeScript CLI's `ollama/probe.ts` asks the same way, and the words
are pinned by `tests/fixtures/w2ops/models.json`.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OllamaProbe:
    #: Whether anything answered at the address.
    ok: bool
    #: The model ids it serves (`llama3.2:latest`), when it answered.
    models: List[str] = field(default_factory=list)


def probe_ollama(base_url: str, timeout: float = 1.5) -> OllamaProbe:
    """GET `<base_url>/models`; a refused, timed out or malformed answer is `ok=False`."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout) as response:  # noqa: S310 - a loopback address the person set
            if getattr(response, "status", 200) >= 400:
                return OllamaProbe(False)
            body = json.loads(response.read().decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001 - not answering is the answer
        return OllamaProbe(False)
    rows = body.get("data") if isinstance(body, dict) else None
    models = [row.get("id") for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)] if isinstance(rows, list) else []
    return OllamaProbe(True, models)


def serves_model(served: List[str], wanted: str) -> bool:
    """Whether a served id counts as the wanted model: the same id, or, for a
    wanted name with no tag, the served name's part before the colon
    (`llama3.2` is served as `llama3.2:latest` or `llama3.2:3b`)."""
    return any(id_ == wanted or (":" not in wanted and id_.split(":", 1)[0] == wanted) for id_ in served)


def ollama_model_check(model: str, base_url: str, probe: OllamaProbe) -> Dict[str, Optional[str]]:
    """The `model` check of `webagents doctor` for an `ollama/<model>` agent,
    from what the probe found: the fixture's words for answering with the
    model, answering without it, and not answering."""
    bare = model.split("/", 1)[1] if "/" in model else model
    if not probe.ok:
        return {
            "status": "fail",
            "detail": f"{model}, but nothing answers at {base_url}",
            "fix": f"Start Ollama (`ollama serve`) and pull the model (`ollama pull {bare}`), or set OLLAMA_BASE_URL",
        }
    if not serves_model(probe.models, bare):
        return {"status": "warn", "detail": f"{model}, at {base_url}, but the model is not pulled", "fix": f"`ollama pull {bare}`"}
    return {"status": "ok", "detail": f"{model}, at {base_url}", "fix": None}
