"""Full rebuilds may reuse indexing calls without inheriting stale answers."""

from __future__ import annotations

import json
from pathlib import Path

from deeptutor.services.rag.pipelines.lightrag.cache_reuse import inherit_index_cache
from deeptutor.services.rag.pipelines.lightrag.engine import workspace_for


def _policy(fingerprint: str) -> dict:
    return {
        "schema_version": 2,
        "policy": "pinned",
        "extract": {"fingerprint": fingerprint},
        "vlm": {"mode": "disabled"},
    }


def _cache(root: Path, entries: dict, *, nested: bool = True) -> Path:
    target = root / workspace_for(root) if nested else root
    target.mkdir(parents=True, exist_ok=True)
    path = target / "kv_store_llm_response_cache.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _entry(kind: str, value: str) -> dict:
    return {"cache_type": kind, "return": value, "original_prompt": "content bound"}


def test_rebuild_inherits_only_content_bound_index_responses(tmp_path: Path) -> None:
    old = tmp_path / "version-1"
    old.mkdir()
    (old / "meta.json").write_text(json.dumps({"indexing_policy": _policy("same")}), "utf-8")
    entries = {
        f"default:extract:{'a' * 32}": _entry("extract", "entities"),
        f"default:analysis:{'b' * 32}": _entry("analysis", "image description"),
        f"default:summary:{'c' * 32}": _entry("summary", "entity summary"),
        f"default:smartheading:{'d' * 32}": _entry("smartheading", "heading"),
        f"global:query:{'e' * 32}": _entry("query", "stale answer"),
        f"global:keywords:{'f' * 32}": _entry("keywords", "stale keywords"),
        "bad": _entry("extract", "bad key"),
        f"default:extract:{'1' * 32}": _entry("query", "mismatched type"),
    }
    _cache(old, entries)
    target = tmp_path / "version-2"
    target.mkdir()

    assert inherit_index_cache(tmp_path, target, _policy("same")) is True

    copied = json.loads(_cache_path(target).read_text(encoding="utf-8"))
    assert len(copied) == 4
    assert {value["cache_type"] for value in copied.values()} == {
        "extract",
        "analysis",
        "summary",
        "smartheading",
    }
    assert inherit_index_cache(tmp_path, target, _policy("same")) is False


def _cache_path(root: Path) -> Path:
    return root / workspace_for(root) / "kv_store_llm_response_cache.json"


def test_interrupted_latest_version_without_meta_keeps_its_cache(tmp_path: Path) -> None:
    published = tmp_path / "version-1"
    published.mkdir()
    (published / "meta.json").write_text(
        json.dumps({"indexing_policy": _policy("same")}), encoding="utf-8"
    )
    _cache(published, {f"default:extract:{'a' * 32}": _entry("extract", "old")})
    interrupted = tmp_path / "version-2"
    _cache(interrupted, {f"default:extract:{'b' * 32}": _entry("extract", "new")})
    target = tmp_path / "version-3"
    target.mkdir()

    assert inherit_index_cache(tmp_path, target, _policy("same")) is True
    assert next(iter(json.loads(_cache_path(target).read_text()).values()))["return"] == "new"


def test_matching_nested_role_policy_beats_newer_mismatch(tmp_path: Path) -> None:
    matching = tmp_path / "version-1"
    matching.mkdir()
    (matching / "meta.json").write_text(
        json.dumps({"indexing_policy": _policy("same")}), encoding="utf-8"
    )
    _cache(matching, {f"default:extract:{'a' * 32}": _entry("extract", "matching")}, nested=False)
    different = tmp_path / "version-2"
    different.mkdir()
    (different / "meta.json").write_text(
        json.dumps({"indexing_policy": _policy("different")}), encoding="utf-8"
    )
    _cache(different, {f"default:extract:{'b' * 32}": _entry("extract", "different")})
    target = tmp_path / "version-3"
    target.mkdir()

    assert inherit_index_cache(tmp_path, target, _policy("same")) is True
    assert next(iter(json.loads(_cache_path(target).read_text()).values()))["return"] == (
        "matching"
    )
