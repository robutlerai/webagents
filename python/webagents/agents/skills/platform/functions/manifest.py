"""
Function manifest schema (Python — mirrors the TS SDK).
"""

from __future__ import annotations
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

FunctionRuntimeId = Literal["js-v1", "python-pyodide-v1", "wasm-v1"]


class CodeRefContent(BaseModel):
    kind: Literal["content"] = "content"
    contentId: str


class CodeRefHttps(BaseModel):
    kind: Literal["https"] = "https"
    url: str


class CodeRefFile(BaseModel):
    """Localhost-only — rejected by the cloud validator."""
    kind: Literal["file"] = "file"
    path: str


class CodeRefInline(BaseModel):
    kind: Literal["inline"] = "inline"
    source: str


class CodeRefInlineB64(BaseModel):
    kind: Literal["inlineB64"] = "inlineB64"
    sourceB64: str


CodeRef = Union[CodeRefContent, CodeRefHttps, CodeRefFile, CodeRefInline, CodeRefInlineB64]


class FolderBinding(BaseModel):
    alias: str
    contentId: str
    permissions: Literal["ro", "rw"] = "ro"


class FunctionPermissions(BaseModel):
    fetch: Optional[List[str]] = None
    secrets: Optional[List[str]] = None
    kv: Optional[Literal["ro", "rw"]] = None
    portal: Optional[List[str]] = None
    folders: Optional[List[FolderBinding]] = None
    rawBody: Optional[bool] = None


class FunctionLimits(BaseModel):
    wallMs: Optional[int] = None
    cpuMs: Optional[int] = None
    memoryMb: Optional[int] = None
    ingressBytes: Optional[int] = None
    egressBytes: Optional[int] = None


class FunctionManifest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    runtime: FunctionRuntimeId = "js-v1"
    entrypoint: Optional[str] = "handler"
    code: Optional[CodeRef] = None
    permissions: Optional[FunctionPermissions] = None
    limits: Optional[FunctionLimits] = None
    bundleSha256: Optional[str] = None
    parameters: Optional[dict] = None
    type: Optional[Literal["http", "websocket", "tool", "cron", "function"]] = None
