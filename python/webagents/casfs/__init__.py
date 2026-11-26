"""CASFS: project data as a tree of content-addressed files, read by path and
verified by hash, with mounts of other projects and published PDKs."""

from .backends import (
    DirChunkCache,
    LocalDirBackend,
    LocalHeads,
    MemoryBackend,
    MemoryChunkCache,
    PlatformApi,
    PlatformBackend,
    default_cache_dir,
    local_package_backend,
)
from .core import MAX_MOUNT_DEPTH, CasError, canon, check_dir_manifest, pdk_mount_path, sha256_hex
from .fs import CasFs, CasHandle
from .project import CasProject

__all__ = [
    "CasError", "CasFs", "CasHandle", "CasProject", "DirChunkCache", "LocalDirBackend", "LocalHeads", "MAX_MOUNT_DEPTH",
    "MemoryBackend", "MemoryChunkCache", "PlatformApi", "PlatformBackend", "canon", "check_dir_manifest",
    "default_cache_dir", "local_package_backend", "pdk_mount_path", "sha256_hex",
]
