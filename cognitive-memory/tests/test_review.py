"""Tests for the Reviewer approval workflow."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.schemas.reasoning import (
    MemoryScope,
    NodeType,
    ReasoningDAG,
    ReasoningNode,
)


@pytest.fixture
def fake_embed():
    def _embed(self, texts):
        return [[float(i) / 10 + 0.01 * j for j in range(8)] for i in range(len(texts))]
    with patch("src.store.vector_store.MemoryStore.embed", _embed):
        yield


@pytest.fixture
def store(tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    return MemoryStore(
        local_path=str(tmp_path / "local"),
        shared_path=str(tmp_path / "shared"),
    )


@pytest.fixture
def reviewer(store, tmp_path):
    from src.sync.review import Reviewer
    from src.sync.sync import SyncLog
    log = SyncLog(db_path=str(tmp_path / "sync.db"))
    return Reviewer(store=store, log=log, reviewer_name="bob")


def _stage(store, project_id, n_nodes=3, scope=MemoryScope.PROJECT):
    """Helper: store nodes locally and push them to staging."""
    dag = ReasoningDAG(
        session_id=f"sess-{project_id}",
        nodes=[
            ReasoningNode(
                node_id=f"n{i}",
                node_type=NodeType.HYPOTHESIS,
                summary=f"hypothesis {project_id} {i}",
                evidence="...",
                message_range=(i, i + 1),
                confidence=0.7,
                scope=scope,
            )
            for i in range(n_nodes)
        ],
        edges=[],
    )
    store.store_dag(dag, project_id)
    unpushed = store.get_unpushed_nodes(project_id)
    store.stage_to_shared(unpushed, developer="alice")


# ── Reviewer init ────────────────────────────────────────────────────────


def test_reviewer_requires_shared_store(tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    from src.sync.review import Reviewer
    local_only = MemoryStore(local_path=str(tmp_path / "local"))
    with pytest.raises(RuntimeError):
        Reviewer(store=local_only)


def test_pending_count(store, reviewer):
    _stage(store, "proj1", n_nodes=4)
    assert reviewer.pending_count("proj1") == 4
    assert reviewer.pending_count() == 4


def test_pending_count_zero_when_empty(store, reviewer):
    assert reviewer.pending_count("proj1") == 0


# ── Approve ──────────────────────────────────────────────────────────────


def test_approve_all_promotes_to_main(store, reviewer):
    from src.sync.review import ReviewAction, ReviewDecision

    _stage(store, "proj1", n_nodes=3)

    def always_approve(node, idx, total):
        return ReviewDecision(action=ReviewAction.APPROVE)

    summary = reviewer.review("proj1", always_approve)
    assert summary.approved == 3
    assert summary.rejected == 0
    assert summary.total_pending == 3

    # Staging is empty
    assert reviewer.pending_count("proj1") == 0
    # Shared main has them
    approved = store.list_approved_shared(project_id="proj1")
    assert len(approved) == 3
    assert all(n["approved"] is True for n in approved)
    assert all(n["approved_by"] == "bob" for n in approved)


# ── Reject ───────────────────────────────────────────────────────────────


def test_reject_keeps_in_staging_marked_rejected(store, reviewer):
    from src.sync.review import ReviewAction, ReviewDecision

    _stage(store, "proj1", n_nodes=2)

    def always_reject(node, idx, total):
        return ReviewDecision(action=ReviewAction.REJECT, reason="hallucinated")

    summary = reviewer.review("proj1", always_reject)
    assert summary.rejected == 2
    assert summary.approved == 0

    # Pending count drops to 0 (rejected nodes are filtered out)
    assert reviewer.pending_count("proj1") == 0

    # But staging still contains them, marked rejected
    all_staged = store.list_pending_in_staging("proj1")
    rejected = [n for n in all_staged if n.get("rejected")]
    assert len(rejected) == 2
    assert all(n["rejection_reason"] == "hallucinated" for n in rejected)


# ── Skip ─────────────────────────────────────────────────────────────────


def test_skip_leaves_node_pending(store, reviewer):
    from src.sync.review import ReviewAction, ReviewDecision

    _stage(store, "proj1", n_nodes=3)

    def skip_all(node, idx, total):
        return ReviewDecision(action=ReviewAction.SKIP)

    summary = reviewer.review("proj1", skip_all)
    assert summary.skipped == 3
    assert summary.approved == 0
    # Still pending after skip
    assert reviewer.pending_count("proj1") == 3


# ── Quit ─────────────────────────────────────────────────────────────────


def test_quit_stops_review_loop(store, reviewer):
    from src.sync.review import ReviewAction, ReviewDecision

    _stage(store, "proj1", n_nodes=5)

    call_log = []

    def quit_after_two(node, idx, total):
        call_log.append(idx)
        if idx >= 2:
            return ReviewDecision(action=ReviewAction.QUIT)
        return ReviewDecision(action=ReviewAction.APPROVE)

    summary = reviewer.review("proj1", quit_after_two)
    assert summary.approved == 2
    # The 3rd call quit
    assert len(call_log) == 3
    # The remaining 2 are still pending
    assert reviewer.pending_count("proj1") == 3


# ── Swap scope ───────────────────────────────────────────────────────────


def test_swap_scope_then_approve(store, reviewer):
    from src.sync.review import ReviewAction, ReviewDecision

    _stage(store, "proj1", n_nodes=2, scope=MemoryScope.PROJECT)

    decisions_per_node = {}

    def swap_then_approve(node, idx, total):
        # On the first invocation for this node: swap scope
        # On the second invocation: approve
        nid = node["id"]
        count = decisions_per_node.get(nid, 0)
        decisions_per_node[nid] = count + 1
        if count == 0:
            return ReviewDecision(action=ReviewAction.SWAP_SCOPE)
        return ReviewDecision(action=ReviewAction.APPROVE)

    summary = reviewer.review("proj1", swap_then_approve)
    assert summary.approved == 2
    assert summary.scope_changes == 2

    # All nodes should now be team scope in shared main
    approved = store.list_approved_shared(project_id="proj1", include_team=True)
    assert all(n["scope"] == "team" for n in approved)


# ── Edit summary ─────────────────────────────────────────────────────────


def test_edit_summary_then_approve(store, reviewer):
    from src.sync.review import ReviewAction, ReviewDecision

    _stage(store, "proj1", n_nodes=1)

    state = {"step": 0}

    def edit_then_approve(node, idx, total):
        if state["step"] == 0:
            state["step"] = 1
            return ReviewDecision(
                action=ReviewAction.EDIT_SUMMARY,
                new_summary="Reviewer-edited summary",
            )
        return ReviewDecision(action=ReviewAction.APPROVE)

    summary = reviewer.review("proj1", edit_then_approve)
    assert summary.approved == 1
    assert summary.summary_edits == 1

    approved = store.list_approved_shared(project_id="proj1")
    assert approved[0]["summary"] == "Reviewer-edited summary"


# ── Mixed decisions ──────────────────────────────────────────────────────


def test_mixed_approve_reject_skip(store, reviewer):
    from src.sync.review import ReviewAction, ReviewDecision

    _stage(store, "proj1", n_nodes=4)

    def varied(node, idx, total):
        if idx == 0:
            return ReviewDecision(action=ReviewAction.APPROVE)
        if idx == 1:
            return ReviewDecision(action=ReviewAction.REJECT, reason="bad")
        if idx == 2:
            return ReviewDecision(action=ReviewAction.SKIP)
        return ReviewDecision(action=ReviewAction.APPROVE)

    summary = reviewer.review("proj1", varied)
    assert summary.approved == 2
    assert summary.rejected == 1
    assert summary.skipped == 1


# ── Audit log ────────────────────────────────────────────────────────────


def test_review_writes_audit_log(store, reviewer):
    from src.sync.review import ReviewAction, ReviewDecision

    _stage(store, "proj1", n_nodes=3)

    def approve(node, idx, total):
        return ReviewDecision(action=ReviewAction.APPROVE)

    reviewer.review("proj1", approve)

    last = reviewer.log.last_event("proj1", "approve")
    assert last is not None
    assert last["count"] == 3
    assert last["actor"] == "bob"
