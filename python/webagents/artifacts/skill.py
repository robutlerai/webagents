"""The producer endpoint: ``GET <base>/artifacts/blobs/sha256:<hex>`` serves a
published blob, or a manifest, by hash.

- RFC 9110 ranges: ``Accept-Ranges: bytes``, ``206`` with ``Content-Range``
  for one range, ``416`` past the end, ``200`` for the whole representation;
- RFC 9530 ``Repr-Digest`` of the whole representation on ``200`` and ``206``;
- streamed from disk, one read buffer per transfer, never loaded whole;
- authorised by who signed the request: the handler is scoped to a verified
  caller (``scope="user"``), and the registry decides from the caller's
  principals (``agent:<url>`` as the access skill records a verified
  signature) and, for a set published for a task, from the ``Artifact-Task``
  header. A URL alone grants nothing; the refusals are named;
- bounded: at most ``max_concurrent_transfers`` bodies in flight, refused by
  name (``503 too_many_transfers``) past that.

``HEAD`` answers the same headers with no body.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from starlette.responses import Response, StreamingResponse

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http
from webagents.server.context.context_vars import get_context

from .http_range import parse_range_header, repr_digest
from .manifest import ARTIFACT_SET_MEDIA_TYPE, parse_digest_name
from .registry import ArtifactRegistry

ARTIFACT_BLOB_SUBPATH = "/artifacts/blobs/{digest}"
ARTIFACT_TASK_HEADER = "artifact-task"
DEFAULT_MAX_CONCURRENT_TRANSFERS = 16
TRANSFER_READ_SIZE = 1024 * 1024


def _json(status: int, body: Any, headers: Optional[Dict[str, str]] = None) -> Response:
    base = {"Cache-Control": "no-store"}
    if headers:
        base.update(headers)
    return Response(json.dumps(body), status_code=status, media_type="application/json", headers=base)


def _principals(context: Any) -> List[str]:
    auth = getattr(context, "auth", None) if context is not None else None
    if auth is None or not getattr(auth, "authenticated", False):
        return []
    raw = getattr(auth, "principals", None) or []
    return [p for p in raw if isinstance(p, str)]


class ArtifactsSkill(Skill):
    """Serves published artifacts by hash to verified callers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, registry: Optional[ArtifactRegistry] = None, max_concurrent_transfers: int = DEFAULT_MAX_CONCURRENT_TRANSFERS) -> None:
        super().__init__(config or {}, scope="all")
        self.registry = registry or ArtifactRegistry()
        self.max_transfers = max_concurrent_transfers
        self._in_flight = 0

    @property
    def active_transfers(self) -> int:
        return self._in_flight

    @http(ARTIFACT_BLOB_SUBPATH, method="get", scope="user")
    async def get_blob(self, digest: str = "") -> Response:
        return await self.serve(digest, head_only=False)

    @http(ARTIFACT_BLOB_SUBPATH, method="head", scope="user")
    async def head_blob(self, digest: str = "") -> Response:
        return await self.serve(digest, head_only=True)

    async def serve(self, digest: str, *, head_only: bool, context: Any = None, headers: Optional[Dict[str, str]] = None) -> Response:
        """Serve one digest for the caller of ``context`` (the current request
        context when none is given). ``headers`` are the request's when the
        caller passes them (tests); else they are read off the context's request."""
        context = context if context is not None else get_context()
        request = getattr(context, "request", None) if context is not None else None
        req_headers = headers if headers is not None else (getattr(request, "headers", None) or {})
        sha256 = parse_digest_name(digest)
        if sha256 is None:
            return _json(404, {"error": {"code": "not_found", "message": "Blobs are named sha256:<64 hex>"}})
        principals = _principals(context)
        if not principals:
            return _json(401, {"error": {"code": "unauthorized", "message": "This blob is served to a caller whose signature this agent verified"}})
        task = req_headers.get(ARTIFACT_TASK_HEADER) if hasattr(req_headers, "get") else None
        verdict = self.registry.access(sha256, principals, task)
        if not verdict.ok:
            if verdict.reason in ("task_required", "task_mismatch"):
                message = (
                    f"This set was published for a task; send it in {ARTIFACT_TASK_HEADER}"
                    if verdict.reason == "task_required"
                    else "This set was published for a different task"
                )
                return _json(403, {"error": {"code": verdict.reason, "message": message}})
            return _json(404, {"error": {"code": "not_found", "message": "No such blob for this caller"}})

        manifest = self.registry.get_manifest(sha256)
        blob = None if manifest else self.registry.get_blob(sha256)
        size = len(manifest.canonical) if manifest else blob.size  # type: ignore[union-attr]
        media_type = ARTIFACT_SET_MEDIA_TYPE if manifest else blob.media_type  # type: ignore[union-attr]
        out_headers: Dict[str, str] = {
            "Accept-Ranges": "bytes",
            "Repr-Digest": repr_digest(sha256),
            "ETag": f'"sha256:{sha256}"',
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        }
        parsed = parse_range_header(req_headers.get("range") if hasattr(req_headers, "get") else None, size)
        if parsed[0] == "unsatisfiable":
            out_headers["Content-Range"] = f"bytes */{size}"
            return Response(None, status_code=416, headers=out_headers, media_type=media_type)
        status = 200
        start, end = 0, size - 1
        if parsed[0] == "single":
            status = 206
            start, end = parsed[1], parsed[2]  # type: ignore[misc]
            out_headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = 0 if size == 0 else end - start + 1
        out_headers["Content-Length"] = str(length)
        if head_only or length == 0:
            return Response(None, status_code=status, headers=out_headers, media_type=media_type)

        if self._in_flight >= self.max_transfers:
            return _json(
                503,
                {"error": {"code": "too_many_transfers", "message": f"This agent is carrying {self._in_flight} transfers, its limit is {self.max_transfers}; retry shortly", "limit": self.max_transfers}},
                {"Retry-After": "2"},
            )
        if manifest:
            return Response(manifest.canonical[start:end + 1], status_code=status, headers=out_headers, media_type=media_type)

        self._in_flight += 1
        path = blob.path  # type: ignore[union-attr]

        async def body() -> AsyncIterator[bytes]:
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        piece = f.read(min(TRANSFER_READ_SIZE, remaining))
                        if not piece:
                            break
                        remaining -= len(piece)
                        yield piece
            finally:
                self._in_flight -= 1

        return StreamingResponse(body(), status_code=status, headers=out_headers, media_type=media_type)


__all__ = ["ArtifactsSkill", "ARTIFACT_BLOB_SUBPATH", "ARTIFACT_TASK_HEADER", "DEFAULT_MAX_CONCURRENT_TRANSFERS"]

# Keep the module's imports honest for the static annotation lint.
_ = os
