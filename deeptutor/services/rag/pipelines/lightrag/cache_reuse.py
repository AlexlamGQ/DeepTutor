"""Seed a rebuild with content-bound LightRAG indexing responses (#1577)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

from deeptutor.services.file_io import atomic_write_json

from .engine import workspace_for

logger = logging.getLogger(__name__)

_CACHE_FILE = "kv_store_llm_response_cache.json"
# Query answers and keyword extraction depend on the current graph and must
# never cross a full rebuild. These types are produced from source content by
# the pinned LightRAG SDK and include the model identity in their hash.
_INDEX_TYPES = frozenset({"extract", "summary", "analysis", "smartheading"})
_CACHE_KEY = re.compile(r"^[^:]+:([a-z]+):[0-9a-f]{32,64}$")


def _policy_identity(policy: Any) -> tuple[str, str, str] | None:
    """Read both current nested role fingerprints and older flat policies."""
    if not isinstance(policy, dict):
        return None
    extract = policy.get("extract")
    if isinstance(extract, dict):
        vlm = policy.get("vlm") if isinstance(policy.get("vlm"), dict) else {}
        vlm_snapshot = vlm.get("snapshot") if isinstance(vlm.get("snapshot"), dict) else {}
        fingerprint = extract.get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            return (
                fingerprint,
                str(vlm.get("mode") or "disabled"),
                str(vlm_snapshot.get("fingerprint") or ""),
            )
    fingerprint = policy.get("fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        return (fingerprint, "", "")
    return None


def _read_policy_identity(root: Path) -> tuple[str, str, str] | None:
    try:
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Interrupted versions have no publication marker but may already
        # have a durable cache worth preserving.
        return None
    return _policy_identity(meta.get("indexing_policy")) if isinstance(meta, dict) else None


def _index_entries(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable LightRAG cache %s: %s", path, type(exc).__name__)
        return {}
    if not isinstance(payload, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        match = _CACHE_KEY.fullmatch(key) if isinstance(key, str) else None
        if (
            match
            and match.group(1) in _INDEX_TYPES
            and isinstance(value, dict)
            and value.get("cache_type") == match.group(1)
            and isinstance(value.get("return"), str)
            and value["return"]
        ):
            entries[key] = value
    return entries


def inherit_index_cache(kb_dir: Path, target_root: Path, policy: dict[str, Any]) -> bool:
    """Carry only verified indexing responses into a fresh version workspace.

    The newest interrupted version is considered even without ``meta.json``;
    when older published versions are considered, matching nested role
    fingerprints are preferred. Hash keys still guard against model drift.
    """
    target = target_root / workspace_for(target_root) / _CACHE_FILE
    if target.exists():
        return False
    target_identity = _policy_identity(policy)
    target_version = int(target_root.name.removeprefix("version-"))
    candidates: list[tuple[int, int, Path]] = []
    for root in kb_dir.glob("version-*"):
        if not root.is_dir() or root == target_root:
            continue
        try:
            version = int(root.name.removeprefix("version-"))
        except ValueError:
            continue
        donor_identity = _read_policy_identity(root)
        # The immediately preceding interrupted candidate may contain progress
        # that was never published. Otherwise prefer matching role policies.
        identity_rank = 1
        if donor_identity is None and version == target_version - 1:
            identity_rank = 3
        elif donor_identity and target_identity:
            identity_rank = 2 if donor_identity == target_identity else 0
        paths = [root / _CACHE_FILE, *root.glob(f"deeptutor_*/{_CACHE_FILE}")]
        for path in paths:
            if path.is_file() and not path.is_symlink():
                candidates.append((version, identity_rank, path))
    if not candidates:
        return False

    for _, _, donor in sorted(candidates, key=lambda item: (item[1], item[0]), reverse=True):
        entries = _index_entries(donor)
        if not entries:
            continue
        atomic_write_json(target, entries)
        logger.info("Inherited %d LightRAG indexing cache entries from %s", len(entries), donor)
        return True
    return False


__all__ = ["inherit_index_cache"]
