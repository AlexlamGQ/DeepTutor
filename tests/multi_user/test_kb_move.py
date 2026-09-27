"""One-KB moves retain bytes, assignments, and saved qualified references."""

from __future__ import annotations

import json

from fastapi import BackgroundTasks
import pytest

from deeptutor.multi_user.knowledge_access import (
    current_kb_manager,
    list_visible_knowledge_bases,
    resolve_kb,
)
from deeptutor.services.workspace import ContentWorkspaceService, WorkspaceError
from deeptutor.services.workspace.context import workspace_context
from deeptutor.services.workspace.kb_move import move_kb, preview_kb_move
from deeptutor.services.workspace.knowledge import qualified_kb_id


def _make_kb(name: str, marker: bytes) -> None:
    manager = current_kb_manager()
    folder = manager.base_dir / name
    (folder / "raw").mkdir(parents=True)
    (folder / "raw" / "source.pdf").write_bytes(marker)
    (folder / "version-1" / "index").parent.mkdir()
    (folder / "version-1" / "index").write_bytes(b"index:" + marker)
    manager.config = manager._load_config()
    manager.config.setdefault("knowledge_bases", {})[name] = {
        "path": name,
        "rag_provider": "llamaindex",
        "status": "ready",
        "description": "an atlas",
    }
    manager._save_config()


def test_move_one_kb_into_nonempty_workspace_preserves_bytes_and_references(as_user):
    with as_user("alice"):
        _make_kb("atlas", b"source-pixels")
        source_manager = current_kb_manager()
        source_manager.config = source_manager._load_config()
        source_manager.config["defaults"] = {"default_kb": "atlas"}
        source_manager._save_config()
        service = ContentWorkspaceService()
        destination = service.create_workspace("Research")
        destination_id = destination["workspace_id"]
        with workspace_context(destination_id):
            _make_kb("existing", b"keep-me")
        old_id = qualified_kb_id("atlas")
        new_id = qualified_kb_id("atlas", destination_id)
        consumer = service.create_workspace("Consumer", resources={"knowledge_bases": [old_id]})

        plan = preview_kb_move(old_id, destination_id)
        assert plan["blockers"] == []
        assert plan["files"] == 2
        assert [row["display_name"] for row in plan["assignments"]] == ["Consumer"]

        result = move_kb(old_id, destination_id)
        assert result["target_id"] == new_id
        with workspace_context(destination_id):
            manager = current_kb_manager()
            assert set(manager.list_knowledge_bases()) == {"atlas", "existing"}
            assert manager._load_config()["defaults"]["default_kb"] == "atlas"
            assert (
                manager.base_dir / "atlas" / "raw" / "source.pdf"
            ).read_bytes() == b"source-pixels"
            assert (
                manager.base_dir / "atlas" / "version-1" / "index"
            ).read_bytes() == b"index:source-pixels"
            assert (manager.base_dir / "existing" / "raw" / "source.pdf").read_bytes() == b"keep-me"
        assert "atlas" not in current_kb_manager().list_knowledge_bases()
        assert current_kb_manager()._load_config()["defaults"]["default_kb"] is None
        assert resolve_kb(old_id).id == new_id
        assert resolve_kb("atlas").id == new_id
        moved_in_account = next(row for row in list_visible_knowledge_bases() if row["id"] == new_id)
        assert moved_in_account["provenance_label"] == "Research"
        with workspace_context(consumer["workspace_id"]):
            assert [row["id"] for row in list_visible_knowledge_bases()] == [new_id]
            assert resolve_kb(old_id).base_dir == resolve_kb(new_id).base_dir
        selected = next(
            row for row in service._catalog() if row["workspace_id"] == consumer["workspace_id"]
        )
        assert selected["resources"]["knowledge_bases"] == [new_id]


def test_move_reports_name_conflict_without_touching_either_catalog(as_user):
    with as_user("alice"):
        _make_kb("atlas", b"source")
        service = ContentWorkspaceService()
        destination = service.create_workspace("Research")["workspace_id"]
        with workspace_context(destination):
            _make_kb("atlas", b"destination")
        source_id = qualified_kb_id("atlas")
        assert "already contains" in preview_kb_move(source_id, destination)["blockers"][0]
        with pytest.raises(WorkspaceError, match="already contains"):
            move_kb(source_id, destination)
        assert (
            current_kb_manager().base_dir / "atlas" / "raw" / "source.pdf"
        ).read_bytes() == b"source"
        with workspace_context(destination):
            assert (
                current_kb_manager().base_dir / "atlas" / "raw" / "source.pdf"
            ).read_bytes() == b"destination"


def test_failed_publish_restores_source_destination_and_assignment(as_user, monkeypatch):
    from deeptutor.services.workspace import kb_move

    with as_user("alice"):
        _make_kb("atlas", b"source")
        service = ContentWorkspaceService()
        destination = service.create_workspace("Research")["workspace_id"]
        old_id = qualified_kb_id("atlas")
        consumer = service.create_workspace("Consumer", resources={"knowledge_bases": [old_id]})
        source_config = current_kb_manager().config_file
        original = kb_move.atomic_write_json
        failed = False

        def fail_once(path, payload):
            nonlocal failed
            if path == source_config and not failed:
                failed = True
                raise OSError("injected write failure")
            original(path, payload)

        monkeypatch.setattr(kb_move, "atomic_write_json", fail_once)
        with pytest.raises(OSError, match="injected"):
            move_kb(old_id, destination)
        assert "atlas" in json.loads(source_config.read_text())["knowledge_bases"]
        assert (source_config.parent / "atlas" / "raw" / "source.pdf").read_bytes() == b"source"
        with workspace_context(destination):
            assert "atlas" not in current_kb_manager().list_knowledge_bases()
            assert not (current_kb_manager().base_dir / "atlas").exists()
        row = next(
            row for row in service._catalog() if row["workspace_id"] == consumer["workspace_id"]
        )
        assert row["resources"]["knowledge_bases"] == [old_id]
        assert resolve_kb(old_id).base_dir == source_config.parent


@pytest.mark.asyncio
async def test_create_destination_overrides_library_scope(as_user, monkeypatch):
    from deeptutor.services import config

    monkeypatch.setattr(config, "load_config_with_main", lambda *_args: {})
    from deeptutor.api.routers import knowledge
    from deeptutor.services.workspace.knowledge import library_request

    with as_user("alice"):
        destination = ContentWorkspaceService().create_workspace("Research")["workspace_id"]
        seen = []

        async def fake_create(*_args):
            seen.append(knowledge._current_kb_base_dir())
            return {"name": "new"}

        monkeypatch.setattr(knowledge, "_create_knowledge_base_owned", fake_create)
        token = library_request.set(True)
        try:
            result = await knowledge.create_knowledge_base(
                BackgroundTasks(),
                name="new",
                files=[],
                rag_provider="llamaindex",
                pageindex_mode="",
                search_mode="",
                rel_paths=[],
                indexing_llm="",
                embedding_model="",
                storage_workspace_id=destination,
            )
        finally:
            library_request.reset(token)
        with workspace_context(destination):
            assert seen == [current_kb_manager().base_dir]
        assert result["id"] == qualified_kb_id("new", destination)
