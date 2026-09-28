"""
A2A v1.0 protocol rules shared by the JSON-RPC and HTTP+JSON bindings of
`A2ATransportSkill`, the Python half of `a2a/protocol.ts`: which methods exist
and what they alias, how the version is decided, how a wire message becomes
the agent's run and how the run's output becomes parts.

VERSION RULES, AND WHY THEY DEPART FROM THE SPEC (plan item 1.3, 2026-09-26).
Section 3.6 says a missing `A2A-Version` header MUST be read as 0.3. OpenClaw's
A2A channel sends no header at all and speaks v1.0 (`SendMessage`,
`ROLE_USER`, `returnImmediately`); a2a-python reads that as 0.3 and refuses it
with -32009, which is exactly the interop the plan wants. So the version comes
from the method name when the header is missing: a PascalCase method is 1.0,
a dotted v0.3 alias gets 0.3-tolerant parsing (`kind` on parts, bare roles,
`mimeType`, `file.uri`). The tolerant parse is applied to every request. An
explicit version other than 1.0 is VersionNotSupported, `0.3` included: no
v0.3 client ever sent the header.

ANSWERS ARE ALWAYS v1.0-SHAPED (`{"task": ...}`), for the dotted aliases too.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from webagents.server.core.credential_floor import CREDENTIAL_HEADERS

from .types import A2A_VERSION_HEADER, A2A_VERSIONS_ACCEPTED, A2AError

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

#: v1.0 PascalCase names and the v0.3 dotted aliases, to one operation each.
METHODS: Dict[str, Tuple[str, bool]] = {
    "SendMessage": ("send", False),
    "message/send": ("send", True),
    "SendStreamingMessage": ("stream", False),
    "message/stream": ("stream", True),
    "GetTask": ("get", False),
    "tasks/get": ("get", True),
    "ListTasks": ("list", False),
    "CancelTask": ("cancel", False),
    "tasks/cancel": ("cancel", True),
    "SubscribeToTask": ("subscribe", False),
    "tasks/resubscribe": ("subscribe", True),
    "CreateTaskPushNotificationConfig": ("push_config", False),
    "GetTaskPushNotificationConfig": ("push_config", False),
    "ListTaskPushNotificationConfigs": ("push_config", False),
    "DeleteTaskPushNotificationConfig": ("push_config", False),
    "tasks/pushNotificationConfig/set": ("push_config", True),
    "tasks/pushNotificationConfig/get": ("push_config", True),
    "tasks/pushNotificationConfig/list": ("push_config", True),
    "tasks/pushNotificationConfig/delete": ("push_config", True),
    "GetExtendedAgentCard": ("extended_card", False),
    "agent/getAuthenticatedExtendedCard": ("extended_card", True),
}


def resolve_method(method: Any) -> Optional[Tuple[str, bool]]:
    """`(operation, dotted)` for a JSON-RPC method name, or None for none."""
    return METHODS.get(method) if isinstance(method, str) else None


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def requested_version(headers: Any, query: Any = None) -> Optional[str]:
    """The `A2A-Version` a request carries: the header, else the `?A2A-Version=` query."""
    value = None
    if headers is not None:
        value = headers.get(A2A_VERSION_HEADER) or headers.get(A2A_VERSION_HEADER.lower())
    if value is None and query is not None:
        value = query.get(A2A_VERSION_HEADER) or query.get(A2A_VERSION_HEADER.lower())
    value = value.strip() if isinstance(value, str) else None
    return value or None


def check_version(version: Optional[str]) -> None:
    """Refuse a version we do not serve. A missing version is accepted; the method decides the parse."""
    if version is None:
        return
    if version not in A2A_VERSIONS_ACCEPTED:
        raise A2AError(
            "VERSION_NOT_SUPPORTED",
            f"Protocol version {version} is not supported; this agent serves A2A 1.0",
            {"requested": version, "supported": "1.0"},
        )


# ---------------------------------------------------------------------------
# Inbound message
# ---------------------------------------------------------------------------

CONTEXT_ID_MAX = 256


def _read_role(value: Any) -> str:
    if value in (None, "", "ROLE_USER", "user"):
        return "ROLE_USER"
    if value in ("ROLE_AGENT", "agent", "assistant"):
        return "ROLE_AGENT"
    raise A2AError("INVALID_PARAMS", f"message.role must be ROLE_USER or ROLE_AGENT, not {json.dumps(value)}")


def read_part(raw: Any, index: int) -> Dict[str, Any]:
    """One part, v1.0 or the v0.3 spelling, as a v1.0 part."""
    if not isinstance(raw, dict):
        raise A2AError("INVALID_PARAMS", f"message.parts[{index}] must be an object")
    media_type = raw.get("mediaType") if isinstance(raw.get("mediaType"), str) else raw.get("mimeType") if isinstance(raw.get("mimeType"), str) else None
    filename = raw.get("filename") if isinstance(raw.get("filename"), str) else raw.get("name") if isinstance(raw.get("name"), str) else None
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else None
    extras: Dict[str, Any] = {}
    if media_type:
        extras["mediaType"] = media_type
    if filename:
        extras["filename"] = filename
    if metadata:
        extras["metadata"] = metadata
    if isinstance(raw.get("text"), str):
        return {"text": raw["text"], **extras}
    if isinstance(raw.get("url"), str):
        return {"url": raw["url"], **extras}
    if isinstance(raw.get("raw"), str):
        return {"raw": raw["raw"], **extras}
    file = raw.get("file")
    if isinstance(file, dict):
        file_type = file.get("mimeType") if isinstance(file.get("mimeType"), str) else file.get("mediaType") if isinstance(file.get("mediaType"), str) else media_type
        file_name = file.get("name") if isinstance(file.get("name"), str) else filename
        file_extras: Dict[str, Any] = {}
        if file_type:
            file_extras["mediaType"] = file_type
        if file_name:
            file_extras["filename"] = file_name
        if metadata:
            file_extras["metadata"] = metadata
        for key, out in (("uri", "url"), ("url", "url"), ("bytes", "raw"), ("data", "raw")):
            if isinstance(file.get(key), str):
                return {out: file[key], **file_extras}
    if "data" in raw and raw["data"] is not None:
        return {"data": raw["data"], **extras}
    raise A2AError("INVALID_PARAMS", f"message.parts[{index}] carries none of text, raw, url or data")


def read_message(raw: Any) -> Dict[str, Any]:
    """The `message` of a send, validated and normalised to v1.0. Ids are minted when missing."""
    if not isinstance(raw, dict):
        raise A2AError("INVALID_PARAMS", "params.message is required")
    parts = raw.get("parts")
    if not isinstance(parts, list) or not parts:
        raise A2AError("INVALID_PARAMS", "message.parts must be a non-empty array")
    context_id = raw.get("contextId")
    if context_id is not None and (not isinstance(context_id, str) or not context_id or len(context_id) > CONTEXT_ID_MAX):
        raise A2AError("INVALID_PARAMS", f"message.contextId must be a string of at most {CONTEXT_ID_MAX} characters")
    task_id = raw.get("taskId")
    if task_id is not None and (not isinstance(task_id, str) or not task_id):
        raise A2AError("INVALID_PARAMS", "message.taskId must be a non-empty string")
    message: Dict[str, Any] = {
        "messageId": raw["messageId"] if isinstance(raw.get("messageId"), str) and raw["messageId"] else str(uuid.uuid4()),
        "role": _read_role(raw.get("role")),
        "parts": [read_part(part, index) for index, part in enumerate(parts)],
    }
    if isinstance(context_id, str):
        message["contextId"] = context_id
    if isinstance(task_id, str):
        message["taskId"] = task_id
    if isinstance(raw.get("metadata"), dict):
        message["metadata"] = raw["metadata"]
    if isinstance(raw.get("extensions"), list):
        message["extensions"] = [e for e in raw["extensions"] if isinstance(e, str)]
    if isinstance(raw.get("referenceTaskIds"), list):
        message["referenceTaskIds"] = [e for e in raw["referenceTaskIds"] if isinstance(e, str)]
    return message


def read_configuration(raw: Any) -> Dict[str, Any]:
    """`params.configuration`: push delivery is refused here, before any task exists."""
    if raw is None:
        return {"return_immediately": False, "history_length": None}
    if not isinstance(raw, dict):
        raise A2AError("INVALID_PARAMS", "params.configuration must be an object")
    if raw.get("taskPushNotificationConfig") is not None:
        raise A2AError("PUSH_NOTIFICATION_NOT_SUPPORTED")
    history = raw.get("historyLength")
    history_length = int(history) if isinstance(history, (int, float)) and not isinstance(history, bool) and history >= 0 else None
    return {"return_immediately": raw.get("returnImmediately") is True, "history_length": history_length}


# ---------------------------------------------------------------------------
# Parts to and from the agent's run (OpenAI message dicts)
# ---------------------------------------------------------------------------


def _media_kind(media_type: Optional[str]) -> str:
    t = (media_type or "").lower()
    if t.startswith("image/"):
        return "image"
    if t.startswith("audio/"):
        return "audio"
    if t.startswith("video/"):
        return "video"
    return "file"


def part_to_content(part: Dict[str, Any]) -> Dict[str, Any]:
    """A v1.0 part as one item of the run's message (the TypeScript
    `partToContentItem`): a `text` item, an OpenAI `image_url` item (the
    multimodal form every LLM skill here reads), or a UAMP-shaped media
    descriptor (`file`, `audio`, `video`: the shapes `uamp/types.ts` gives
    them) that `message_to_run_message` puts under `content_items`."""
    if "text" in part:
        return {"type": "text", "text": part["text"]}
    if "data" in part:
        label = f"[data {part['mediaType']}] " if part.get("mediaType") else "[data] "
        return {"type": "text", "text": label + json.dumps(part["data"], ensure_ascii=False)}
    kind = _media_kind(part.get("mediaType"))
    media_type = part.get("mediaType") or "application/octet-stream"
    filename = part.get("filename") or "attachment"
    if "url" in part:
        if kind == "image":
            return {"type": "image_url", "image_url": {"url": part["url"]}}
        if kind == "audio":
            return {"type": "audio", "audio": {"url": part["url"]}}
        if kind == "video":
            return {"type": "video", "video": {"url": part["url"]}}
        return {"type": "file", "file": {"url": part["url"]}, "filename": filename, "mime_type": media_type}
    raw = part.get("raw") or ""
    data_url = f"data:{media_type};base64,{raw}"
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": data_url}}
    if kind == "audio":
        return {"type": "audio", "audio": raw, "format": media_type.split("/", 1)[1] if "/" in media_type else "wav"}
    if kind == "video":
        return {"type": "video", "video": data_url}
    return {"type": "file", "file": data_url, "filename": filename, "mime_type": media_type}


def message_to_run_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """A v1.0 message as one of the run's messages, the TypeScript
    `messageToRunMessage`: the text joined in `content`, with any image
    beside it in the OpenAI multimodal list (how the LLM skills here take an
    image), and every other medium under `content_items`, the platform's own
    key for media descriptors on a message, which the LLM adapters strip
    (`openai/uamp_adapter.py`, `anthropic/uamp_adapter.py`) and a skill may
    read. So a handoff that reads the text sees the text, in both SDKs."""
    items = [part_to_content(p) for p in message["parts"]]
    role = "assistant" if message.get("role") == "ROLE_AGENT" else "user"
    texts = [i["text"] for i in items if i["type"] == "text"]
    images = [i for i in items if i["type"] == "image_url"]
    media = [i for i in items if i["type"] not in ("text", "image_url")]
    content: Any = "\n".join(texts)
    if images:
        content = [{"type": "text", "text": content}, *images] if texts else images
    out: Dict[str, Any] = {"role": role, "content": content}
    if media:
        out["content_items"] = media
    return out


def output_to_parts(content: Any) -> List[Dict[str, Any]]:
    """The run's output as v1.0 parts: the text, and any image the completions shape carried."""
    if isinstance(content, list):
        parts: List[Dict[str, Any]] = []
        texts = [i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text"]
        parts.append({"text": "".join(texts)})
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url") or ""
                match = re.match(r"^data:([^;,]+)(?:;base64)?,(.*)$", url, re.S)
                if match:
                    parts.append({"raw": match.group(2), "mediaType": match.group(1)})
                elif url:
                    parts.append({"url": url, "mediaType": "image/png"})
        return parts
    return [{"text": content if isinstance(content, str) else ""}]


def reply_text(task: Dict[str, Any]) -> str:
    """The text a task's output carries, artifacts first, then the status message (what Hermes reads)."""
    texts = [p["text"] for a in task.get("artifacts", []) for p in a.get("parts", []) if isinstance(p.get("text"), str)]
    if texts:
        return "".join(texts)
    message = (task.get("status") or {}).get("message") or {}
    return "".join(p["text"] for p in message.get("parts", []) if isinstance(p.get("text"), str))


# ---------------------------------------------------------------------------
# The caller
# ---------------------------------------------------------------------------


def caller_key(auth: Any, headers: Any) -> str:
    """The key a task is owned under. A caller the agent verified is its
    principal (the access block's, else the auth skill's user or agent id);
    one it could not verify is the credential it presented, hashed, so the
    same bearer reads back its own tasks and nobody else's. No credential at
    all is the anonymous key, which the floor keeps off every task route."""
    if auth is not None and getattr(auth, "authenticated", False):
        principals = getattr(auth, "principals", None) or []
        principal = next((p for p in principals if isinstance(p, str) and p), None)
        candidate = principal or getattr(auth, "user_id", None) or getattr(auth, "agent_id", None)
        if isinstance(candidate, str) and candidate:
            return f"id:{candidate}"
    credential = None
    if headers is not None:
        for name in CREDENTIAL_HEADERS:
            value = headers.get(name)
            if value:
                credential = value
                break
        if credential is None:
            credential = headers.get("signature")
    if not credential:
        return "anonymous"
    return "cred:" + hashlib.sha256(str(credential).encode("utf-8")).hexdigest()


def now_iso() -> str:
    """ISO 8601 UTC with `Z` and milliseconds, as the spec's timestamps are
    written (one clock read: two would straddle a second boundary)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def b64url_no_pad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
