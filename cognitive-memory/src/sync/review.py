"""Review and approval workflow for staged shared-store nodes.

The Reviewer class is the headless, testable core: it iterates over
pending nodes and applies decisions. The interactive CLI in src/cli.py
wraps this with a rich UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from ..store.vector_store import MemoryStore
from .sync import SyncLog


class ReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    SWAP_SCOPE = "swap_scope"
    EDIT_SUMMARY = "edit_summary"
    SKIP = "skip"
    QUIT = "quit"


@dataclass
class ReviewDecision:
    """A reviewer's decision about one staged node."""
    action: ReviewAction
    new_scope: str | None = None        # for SWAP_SCOPE
    new_summary: str | None = None       # for EDIT_SUMMARY
    reason: str = ""                      # for REJECT


@dataclass
class ReviewSummary:
    """Result of running a full review session."""
    project_id: str | None
    total_pending: int = 0
    approved: int = 0
    rejected: int = 0
    skipped: int = 0
    scope_changes: int = 0
    summary_edits: int = 0

    @property
    def text(self) -> str:
        return (
            f"Reviewed {self.total_pending} nodes: "
            f"{self.approved} approved, {self.rejected} rejected, "
            f"{self.skipped} skipped"
        )


class Reviewer:
    """Headless review engine — applies decisions to staged nodes."""

    def __init__(
        self,
        store: MemoryStore,
        log: SyncLog | None = None,
        reviewer_name: str = "",
    ):
        if not store.has_shared:
            raise RuntimeError("Reviewer requires a shared-mode MemoryStore")
        self.store = store
        self.log = log or SyncLog()
        self.reviewer_name = reviewer_name

    def pending_count(self, project_id: str | None = None) -> int:
        """Number of nodes awaiting review (non-rejected, non-approved)."""
        pending = self.store.list_pending_in_staging(project_id)
        return sum(1 for p in pending if not p.get("rejected"))

    def list_pending(self, project_id: str | None = None) -> list[dict[str, Any]]:
        """All pending (non-rejected) nodes for review."""
        pending = self.store.list_pending_in_staging(project_id)
        return [p for p in pending if not p.get("rejected")]

    def review(
        self,
        project_id: str | None,
        decide: Callable[[dict[str, Any], int, int], ReviewDecision],
    ) -> ReviewSummary:
        """Iterate over pending nodes and apply decisions.

        Args:
            project_id: optional project filter
            decide: callable invoked for each pending node — receives
                (node_dict, current_index, total) and returns a ReviewDecision.

        The decide callback is the seam where the interactive CLI plugs in.
        Tests use a stub callback that returns canned decisions.
        """
        pending = self.list_pending(project_id)
        summary = ReviewSummary(project_id=project_id, total_pending=len(pending))

        # Group decisions before applying so we can do batched promote/reject calls
        approve_ids: list[str] = []
        scope_overrides: dict[str, str] = {}
        summary_overrides: dict[str, str] = {}
        reject_ids: list[str] = []
        reject_reasons: dict[str, str] = {}

        for i, node in enumerate(pending):
            decision = decide(node, i, len(pending))

            if decision.action == ReviewAction.QUIT:
                break

            if decision.action == ReviewAction.SKIP:
                summary.skipped += 1
                continue

            nid = node["id"]

            # SWAP_SCOPE and EDIT_SUMMARY are *modifiers* — they change the
            # node's pending state but the reviewer still has to approve or
            # reject afterward. We re-loop on the same node by stacking
            # the modification and re-invoking decide().
            while decision.action in (ReviewAction.SWAP_SCOPE, ReviewAction.EDIT_SUMMARY):
                if decision.action == ReviewAction.SWAP_SCOPE:
                    new_scope = decision.new_scope or (
                        "team" if node.get("scope") == "project" else "project"
                    )
                    node["scope"] = new_scope
                    scope_overrides[nid] = new_scope
                    summary.scope_changes += 1
                elif decision.action == ReviewAction.EDIT_SUMMARY:
                    if decision.new_summary is not None:
                        node["summary"] = decision.new_summary
                        summary_overrides[nid] = decision.new_summary
                        summary.summary_edits += 1
                decision = decide(node, i, len(pending))

            if decision.action == ReviewAction.APPROVE:
                approve_ids.append(nid)
                summary.approved += 1
            elif decision.action == ReviewAction.REJECT:
                reject_ids.append(nid)
                reject_reasons[nid] = decision.reason
                summary.rejected += 1
            elif decision.action == ReviewAction.SKIP:
                summary.skipped += 1
            elif decision.action == ReviewAction.QUIT:
                break

        # Apply decisions in batches
        if approve_ids:
            self.store.promote_from_staging(
                approve_ids,
                approver=self.reviewer_name,
                scope_overrides={k: v for k, v in scope_overrides.items() if k in approve_ids},
                summary_overrides={k: v for k, v in summary_overrides.items() if k in approve_ids},
            )
            self.log.record(
                project_id=project_id or "all",
                action="approve",
                count=len(approve_ids),
                actor=self.reviewer_name,
                detail=f"approved {len(approve_ids)} nodes",
            )

        if reject_ids:
            for nid in reject_ids:
                self.store.reject_in_staging(
                    [nid],
                    reviewer=self.reviewer_name,
                    reason=reject_reasons.get(nid, ""),
                )
            self.log.record(
                project_id=project_id or "all",
                action="reject",
                count=len(reject_ids),
                actor=self.reviewer_name,
                detail=f"rejected {len(reject_ids)} nodes",
            )

        return summary
