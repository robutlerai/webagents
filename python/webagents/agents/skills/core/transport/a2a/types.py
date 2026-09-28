"""
A2A v1.0 wire vocabulary and errors (a2aproject/A2A v1.0.1, `a2a.proto`,
rendered as ProtoJSON), the Python half of `a2a/types.ts`.

v1.0 broke the v0.3 wire (plan item 1.3, 2026-09-26): `kind` is gone from
parts and tasks, `mimeType` became `mediaType`, states are SCREAMING_SNAKE,
the `final` flag on stream events is gone and the card's `url` became
`supportedInterfaces`. Everything here is v1.0; `protocol.py` reads the v0.3
spellings tolerantly on the way in and never writes them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

A2A_VERSION = "1.0"
A2A_VERSION_HEADER = "A2A-Version"
A2A_VERSIONS_ACCEPTED = ("1.0", "1.0.0", "1")

TERMINAL_STATES = frozenset(
    {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}
)
INTERRUPTED_STATES = frozenset({"TASK_STATE_INPUT_REQUIRED", "TASK_STATE_AUTH_REQUIRED"})


def is_settled(state: str) -> bool:
    """Whether a task in `state` will not change again on its own."""
    return state in TERMINAL_STATES or state in INTERRUPTED_STATES


A2A_ERROR_DOMAIN = "a2a-protocol.org"
ERROR_INFO_TYPE = "type.googleapis.com/google.rpc.ErrorInfo"

#: name -> (JSON-RPC code, ErrorInfo reason, HTTP status, message)
A2A_ERRORS: Dict[str, tuple] = {
    "PARSE_ERROR": (-32700, "PARSE_ERROR", 400, "Invalid JSON payload"),
    "INVALID_REQUEST": (-32600, "INVALID_REQUEST", 400, "Request payload validation error"),
    "METHOD_NOT_FOUND": (-32601, "METHOD_NOT_FOUND", 404, "Method not found"),
    "INVALID_PARAMS": (-32602, "INVALID_PARAMS", 400, "Invalid parameters"),
    "INTERNAL_ERROR": (-32603, "INTERNAL_ERROR", 500, "Internal error"),
    "TASK_NOT_FOUND": (-32001, "TASK_NOT_FOUND", 404, "Task not found"),
    "TASK_NOT_CANCELABLE": (-32002, "TASK_NOT_CANCELABLE", 400, "Task cannot be canceled"),
    "PUSH_NOTIFICATION_NOT_SUPPORTED": (-32003, "PUSH_NOTIFICATION_NOT_SUPPORTED", 400, "Push Notification is not supported"),
    "UNSUPPORTED_OPERATION": (-32004, "UNSUPPORTED_OPERATION", 400, "This operation is not supported"),
    "CONTENT_TYPE_NOT_SUPPORTED": (-32005, "CONTENT_TYPE_NOT_SUPPORTED", 400, "Incompatible content types"),
    "INVALID_AGENT_RESPONSE": (-32006, "INVALID_AGENT_RESPONSE", 500, "Invalid agent response"),
    "EXTENDED_AGENT_CARD_NOT_CONFIGURED": (-32007, "EXTENDED_AGENT_CARD_NOT_CONFIGURED", 400, "Extended agent card is not configured"),
    "EXTENSION_SUPPORT_REQUIRED": (-32008, "EXTENSION_SUPPORT_REQUIRED", 400, "A required extension is not supported"),
    "VERSION_NOT_SUPPORTED": (-32009, "VERSION_NOT_SUPPORTED", 400, "Protocol version not supported"),
}


def rpc_status_name(http: int) -> str:
    """`google.rpc.Code` names for the REST error envelope, by HTTP status."""
    return {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        409: "ABORTED",
        501: "UNIMPLEMENTED",
    }.get(http, "INTERNAL" if http >= 500 else "UNKNOWN")


class A2AError(Exception):
    """A protocol error, carrying everything both bindings need to answer it."""

    def __init__(self, name: str, message: Optional[str] = None, metadata: Optional[Dict[str, str]] = None):
        code, reason, http, default = A2A_ERRORS[name]
        super().__init__(message or default)
        self.code = code
        self.reason = reason
        self.http = http
        self.metadata = dict(metadata or {})

    def error_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"@type": ERROR_INFO_TYPE, "reason": self.reason, "domain": A2A_ERROR_DOMAIN}
        if self.metadata:
            info["metadata"] = self.metadata
        return info

    def json_rpc(self) -> Dict[str, Any]:
        return {"code": self.code, "message": str(self), "data": [self.error_info()]}

    def rest(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.http,
                "status": rpc_status_name(self.http),
                "message": str(self),
                "details": [self.error_info()],
            }
        }


def task_view(task: Dict[str, Any], history_length: Optional[int] = None, include_artifacts: bool = True) -> Dict[str, Any]:
    """A task as a response carries it: history trimmed when asked, artifacts optional."""
    history: List[Dict[str, Any]] = task.get("history", [])
    if history_length is not None and history_length >= 0:
        history = history[-history_length:] if history_length else []
    view: Dict[str, Any] = {
        "id": task["id"],
        "contextId": task["contextId"],
        "status": task["status"],
        "artifacts": task.get("artifacts", []) if include_artifacts else [],
        "history": history,
    }
    if task.get("metadata"):
        view["metadata"] = task["metadata"]
    return view
