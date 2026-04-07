"""Consolidate reasoning nodes into a CognitiveProfile via clustering + LLM."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import anthropic
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..schemas.memory import (
    ArchitecturalInsight,
    CognitiveProfile,
    DiagnosticStrategy,
    Pitfall,
)
from ..schemas.reasoning import MemoryScope
from ..store.vector_store import MemoryStore

_MODEL = "claude-sonnet-4-6"
_MIN_CLUSTER_SIZE = 2      # clusters smaller than this are skipped
_MAX_NODES_PER_CLUSTER = 6 # cap nodes sent to LLM per cluster

_CLASSIFY_SYSTEM = """\
You are an expert at distilling coding knowledge from AI agent session logs.
"""

_CLASSIFY_PROMPT = """\
These reasoning fragments come from coding sessions on the same project. \
Determine what type of knowledge they represent and produce a structured summary.

Fragments:
{fragments}

Classify as ONE of:
- ARCHITECTURAL_INSIGHT: Something structural about how this codebase works
- PITFALL: A recurring problem or trap that agents fall into
- DIAGNOSTIC_STRATEGY: A debugging/investigation approach that proved effective

Also classify the SCOPE of this knowledge:
- "project": specific to THIS project's architecture, schema, configuration, or
  conventions. The insight only applies to this codebase. (e.g., "Pydantic models
  in this repo must be registered in models/__init__.py for Alembic to detect them")
- "team": general technical knowledge that applies across ANY project using the
  same tools or frameworks. (e.g., "Pydantic v2 requires migrating from class
  Config to model_config = ConfigDict(...)")

Then respond with JSON only (no markdown):
{{
  "type": "ARCHITECTURAL_INSIGHT" | "PITFALL" | "DIAGNOSTIC_STRATEGY",
  "scope": "project" | "team",

  // If ARCHITECTURAL_INSIGHT:
  "component": "which component or subsystem",
  "insight": "what was learned",
  "confidence": 0.0-1.0,

  // If PITFALL:
  "description": "what the pitfall is",
  "severity": "low" | "medium" | "high",
  "resolution_strategy": "how to avoid or fix it (or null)",

  // If DIAGNOSTIC_STRATEGY:
  "trigger": "when to apply this strategy",
  "steps": ["step 1", "step 2", ...],
  "success_rate": 0.0-1.0
}}"""

_KEY_PATTERNS_PROMPT = """\
Given these node summaries from coding sessions on one project, \
identify 3-5 recurring patterns (things that happen often) and \
3-5 anti-patterns (approaches that consistently fail or cause problems).

Summaries:
{summaries}

Respond with JSON only:
{{
  "key_patterns": ["pattern 1", ...],
  "anti_patterns": ["anti-pattern 1", ...]
}}"""


def _cluster_nodes(
    nodes: list[dict[str, Any]],
    distance_threshold: float = 0.4,
) -> list[list[dict[str, Any]]]:
    """Cluster nodes by embedding similarity. Returns list of clusters."""
    if len(nodes) < 2:
        return [nodes] if nodes else []

    embeddings = np.array([n["embedding"] for n in nodes], dtype=np.float32)

    # Normalise for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    clusters: dict[int, list[dict]] = {}
    for node, label in zip(nodes, labels):
        clusters.setdefault(int(label), []).append(node)
    return list(clusters.values())


async def _classify_cluster(
    client: anthropic.AsyncAnthropic,
    cluster: list[dict[str, Any]],
    session_ids: list[str],
) -> dict[str, Any] | None:
    """Ask the LLM to classify and summarise a cluster of related nodes."""
    sample = cluster[:_MAX_NODES_PER_CLUSTER]
    fragments = "\n".join(
        f"- [{n.get('node_type', '?')}] {n['summary']}" for n in sample
    )
    prompt = _CLASSIFY_PROMPT.format(fragments=fragments)

    try:
        resp = await client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=_CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        data["source_sessions"] = list(set(session_ids))
        return data
    except Exception as e:
        return None


async def _extract_patterns(
    client: anthropic.AsyncAnthropic,
    all_nodes: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Extract key patterns and anti-patterns from all node summaries."""
    summaries = "\n".join(f"- {n['summary']}" for n in all_nodes[:60])
    prompt = _KEY_PATTERNS_PROMPT.format(summaries=summaries)

    try:
        resp = await client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=_CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return data.get("key_patterns", []), data.get("anti_patterns", [])
    except Exception:
        return [], []


class ProfileBuilder:
    """Consolidate stored reasoning nodes into a CognitiveProfile."""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def build_profile(self, project_id: str, store: MemoryStore) -> CognitiveProfile:
        import asyncio

        # 1. Fetch all nodes for this project
        all_nodes = store.get_all_nodes(project_id)
        if not all_nodes:
            return CognitiveProfile(
                project_id=project_id,
                last_updated=datetime.now(timezone.utc),
                session_count=0,
            )

        # 2. Count unique sessions
        session_ids = list({n.get("session_id", "") for n in all_nodes})

        # 3. Cluster by semantic similarity
        clusters = _cluster_nodes(all_nodes)

        # 4. Classify each cluster concurrently
        tasks = [
            _classify_cluster(
                self.client,
                cluster,
                [n.get("session_id", "") for n in cluster],
            )
            for cluster in clusters
            if len(cluster) >= _MIN_CLUSTER_SIZE
        ]
        classifications = await asyncio.gather(*tasks)

        # 5. Extract patterns from all nodes
        key_patterns, anti_patterns = await _extract_patterns(self.client, all_nodes)

        # 6. Assemble profile
        insights: list[ArchitecturalInsight] = []
        pitfalls: list[Pitfall] = []
        strategies: list[DiagnosticStrategy] = []

        for data in classifications:
            if data is None:
                continue
            ctype = data.get("type", "")
            srcs = data.get("source_sessions", [])
            scope_str = str(data.get("scope", "project")).lower()
            try:
                scope = MemoryScope(scope_str)
            except ValueError:
                scope = MemoryScope.PROJECT

            if ctype == "ARCHITECTURAL_INSIGHT":
                insights.append(ArchitecturalInsight(
                    component=data.get("component", "unknown"),
                    insight=data.get("insight", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    scope=scope,
                ))
            elif ctype == "PITFALL":
                pitfalls.append(Pitfall(
                    description=data.get("description", ""),
                    frequency=len(srcs),
                    severity=data.get("severity", "medium"),
                    resolution_strategy=data.get("resolution_strategy"),
                    scope=scope,
                ))
            elif ctype == "DIAGNOSTIC_STRATEGY":
                strategies.append(DiagnosticStrategy(
                    trigger=data.get("trigger", ""),
                    steps=data.get("steps", []),
                    success_rate=float(data.get("success_rate", 0.5)),
                    source_sessions=srcs,
                    scope=scope,
                ))

        # Sort pitfalls by severity
        _sev = {"high": 0, "medium": 1, "low": 2}
        pitfalls.sort(key=lambda p: _sev.get(p.severity, 1))

        profile = CognitiveProfile(
            project_id=project_id,
            last_updated=datetime.now(timezone.utc),
            architectural_insights=insights,
            pitfalls=pitfalls,
            diagnostic_strategies=strategies,
            key_patterns=key_patterns,
            anti_patterns=anti_patterns,
            session_count=len(session_ids),
        )

        # 7. Persist
        store.save_profile(profile)
        return profile
