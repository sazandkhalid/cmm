"""Tests for memory scope (project vs team) classification and persistence."""
from __future__ import annotations

import pytest

from src.schemas.memory import ArchitecturalInsight, DiagnosticStrategy, Pitfall
from src.schemas.reasoning import (
    MemoryScope,
    NodeType,
    ReasoningDAG,
    ReasoningEdge,
    ReasoningNode,
)


# ── Schema defaults ──────────────────────────────────────────────────────


def test_reasoning_node_defaults_to_project_scope():
    n = ReasoningNode(
        node_id="x",
        node_type=NodeType.HYPOTHESIS,
        summary="...",
        evidence="...",
        message_range=(0, 1),
    )
    assert n.scope == MemoryScope.PROJECT


def test_reasoning_node_accepts_team_scope():
    n = ReasoningNode(
        node_id="x",
        node_type=NodeType.HYPOTHESIS,
        summary="Pydantic v2 migration",
        evidence="...",
        message_range=(0, 1),
        scope=MemoryScope.TEAM,
    )
    assert n.scope == MemoryScope.TEAM


def test_pitfall_defaults_to_project_scope():
    p = Pitfall(description="...")
    assert p.scope == MemoryScope.PROJECT


def test_pitfall_team_scope():
    p = Pitfall(description="Pydantic v2 ConfigDict", scope=MemoryScope.TEAM)
    assert p.scope == MemoryScope.TEAM


def test_architectural_insight_defaults_to_project():
    i = ArchitecturalInsight(component="db", insight="...")
    assert i.scope == MemoryScope.PROJECT


def test_diagnostic_strategy_defaults_to_project():
    s = DiagnosticStrategy(trigger="...", steps=["a", "b"])
    assert s.scope == MemoryScope.PROJECT


def test_memory_scope_enum_values():
    assert MemoryScope.PROJECT.value == "project"
    assert MemoryScope.TEAM.value == "team"


def test_memory_scope_from_string():
    assert MemoryScope("project") == MemoryScope.PROJECT
    assert MemoryScope("team") == MemoryScope.TEAM


# ── Round-trip through Pydantic JSON ─────────────────────────────────────


def test_node_serializes_scope():
    n = ReasoningNode(
        node_id="x",
        node_type=NodeType.SOLUTION,
        summary="...",
        evidence="...",
        message_range=(0, 1),
        scope=MemoryScope.TEAM,
    )
    data = n.model_dump()
    assert data["scope"] == "team"


def test_node_deserializes_scope():
    data = {
        "node_id": "x",
        "node_type": "solution",
        "summary": "...",
        "evidence": "...",
        "message_range": [0, 1],
        "confidence": 0.8,
        "scope": "team",
    }
    n = ReasoningNode.model_validate(data)
    assert n.scope == MemoryScope.TEAM


# ── DAG with mixed-scope nodes ───────────────────────────────────────────


def test_dag_carries_mixed_scopes():
    project_node = ReasoningNode(
        node_id="a", node_type=NodeType.HYPOTHESIS,
        summary="check models/__init__.py registration",
        evidence="...", message_range=(0, 2),
        scope=MemoryScope.PROJECT,
    )
    team_node = ReasoningNode(
        node_id="b", node_type=NodeType.DISCOVERY,
        summary="Pydantic v2 deprecates class Config",
        evidence="...", message_range=(2, 4),
        scope=MemoryScope.TEAM,
    )
    dag = ReasoningDAG(
        session_id="s1",
        nodes=[project_node, team_node],
        edges=[ReasoningEdge(source_id="a", target_id="b", relationship="led_to")],
    )
    scopes = sorted(n.scope.value for n in dag.nodes)
    assert scopes == ["project", "team"]
