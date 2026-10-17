"""Streaming hashes for artifact files: the whole-file SHA-256 and, for a
file above the chunk size, the chunk list. A file passes through once in
fixed-size pieces and is never read whole."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import AsyncIterable, Iterable, Optional, Union

from .manifest import CHUNK_SIZE_DEFAULT, ChunkList, is_valid_chunk_size

READ_SIZE = 1024 * 1024


@dataclass
class HashedBytes:
    sha256: str
    size: int
    chunks: Optional[ChunkList] = None


class _Chunker:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.whole = hashlib.sha256()
        self.part = hashlib.sha256()
        self.filled = 0
        self.size = 0
        self.hashes: list[str] = []

    def update(self, data: bytes) -> None:
        self.whole.update(data)
        self.size += len(data)
        off = 0
        while off < len(data):
            take = min(self.chunk_size - self.filled, len(data) - off)
            self.part.update(data[off:off + take])
            self.filled += take
            off += take
            if self.filled == self.chunk_size:
                self.hashes.append(self.part.hexdigest())
                self.part = hashlib.sha256()
                self.filled = 0

    def finish(self) -> HashedBytes:
        if self.filled > 0:
            self.hashes.append(self.part.hexdigest())
        out = HashedBytes(sha256=self.whole.hexdigest(), size=self.size)
        if self.size > self.chunk_size:
            out.chunks = ChunkList(chunk_size=self.chunk_size, hashes=self.hashes)
        return out


def hash_bytes_iter(source: Iterable[Union[bytes, bytearray, memoryview]], chunk_size: int = CHUNK_SIZE_DEFAULT) -> HashedBytes:
    if not is_valid_chunk_size(chunk_size):
        raise ValueError(f"invalid chunk size {chunk_size}")
    chunker = _Chunker(chunk_size)
    for piece in source:
        chunker.update(bytes(piece))
    return chunker.finish()


async def hash_bytes_aiter(source: AsyncIterable[Union[bytes, bytearray, memoryview]], chunk_size: int = CHUNK_SIZE_DEFAULT) -> HashedBytes:
    if not is_valid_chunk_size(chunk_size):
        raise ValueError(f"invalid chunk size {chunk_size}")
    chunker = _Chunker(chunk_size)
    async for piece in source:
        chunker.update(bytes(piece))
    return chunker.finish()


def hash_file(path: Union[str, os.PathLike], chunk_size: int = CHUNK_SIZE_DEFAULT) -> HashedBytes:
    """Hash a file on disk, streaming."""
    st = os.stat(path)
    if not os.path.isfile(path):
        raise ValueError(f"{path} is not a regular file")

    def pieces() -> Iterable[bytes]:
        with open(path, "rb") as f:
            while True:
                piece = f.read(READ_SIZE)
                if not piece:
                    break
                yield piece

    hashed = hash_bytes_iter(pieces(), chunk_size)
    if hashed.size != st.st_size:
        raise ValueError(f"{path} changed while it was being hashed ({st.st_size} then {hashed.size} bytes)")
    return hashed


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["HashedBytes", "hash_file", "hash_bytes_iter", "hash_bytes_aiter", "sha256_hex", "READ_SIZE"]
