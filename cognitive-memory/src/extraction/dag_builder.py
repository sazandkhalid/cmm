"""Reasoning DAG extractor — token-budget LLM extraction.

Replaces the previous fixed 5-message windowing with token-budget chunking
based on the Anthropic SDK's count_tokens() API. Each window is filled to
~45% of the model's context window (configurable via CMM_CONTEXT_FILL_RATIO),
which research suggests is the safe utilization threshold before
"lost in the middle" effects degrade extraction quality.

Pipeline:
    1. Filter noise (empty messages, oversized tool dumps)
    2. Pre-count tokens for every message (concurrent)
    3. Pack messages into token-budget windows with token-based overlap
    4. If session fits in one window → single LLM call (ideal)
    5. Otherwise → 2-3 large windows (NOT 30 tiny ones)
    6. Each window classification returns a LIST of nodes (not just one)
    7. LLM-inferred edges between all classified nodes
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import anthropic

from ..schemas.session import MessageRole, NormalizedSession, SessionMessage
from ..schemas.reasoning import NodeType, ReasoningDAG, ReasoningEdge, ReasoningNode

# ── Constants ────────────────────────────────────────────────────────────────

_MODEL = "claude-sonnet-4-5"  # use a model id that count_tokens supports
_CLASSIFICATION_MODEL = "claude-sonnet-4-5"

# Context-budget tuning
MODEL_CONTEXT_WINDOW = 200_000
DEFAULT_FILL_RATIO = 0.45            # research-backed safe utilization
DEFAULT_OUTPUT_RESERVE = 4_000        # max_tokens for the response
DEFAULT_OVERLAP_TOKENS = 750          # trailing overlap between windows
MIN_WINDOW_TOKENS = 2_000             # never produce a window smaller than this

# Noise filtering (same thresholds as before)
_MAX_TOOL_RESULT_CHARS = 500
_NOISE_TRUNCATE_THRESHOLD = 5_000

# Per-window node cap to keep responses bounded
_MAX_NODES_PER_WINDOW = 12

_NOISE_ROLES = {MessageRole.TOOL_RESULT}

# ── Prompts ──────────────────────────────────────────────────────────────────

_CLASSIFICATION_SYSTEM = """\
You are an expert analyst of AI coding agent sessions. You identify the \
distinct reasoning steps an agent takes during a conversation segment.
"""

_CLASSIFICATION_PROMPT = """\
Analyze this segment of a coding agent's session and identify EACH distinct \
reasoning step the agent took. A single segment will usually contain MULTIPLE \
reasoning steps, not just one.

For each step, classify it as ONE of:
- HYPOTHESIS: The agent forms a theory about what might be wrong or how to proceed
- INVESTIGATION: The agent examines code, runs tests, or gathers evidence
- DISCOVERY: The agent finds something unexpected that changes understanding
- PIVOT: The agent explicitly changes approach based on new information
- SOLUTION: The agent reaches a working resolution
- DEAD_END: The agent's current approach fails and is abandoned
- CONTEXT_LOAD: The agent is reading/understanding code without active reasoning

Each message is prefixed with [N] where N is its absolute index in the session.
When you identify a reasoning step, record the message indices that contain it.

Session segment:
{segment}

Respond with JSON only (no markdown, no explanation):
{{
  "nodes": [
    {{
      "node_type": "HYPOTHESIS|INVESTIGATION|DISCOVERY|PIVOT|SOLUTION|DEAD_END|CONTEXT_LOAD",
      "summary": "1-2 sentence description of what the agent is doing and why",
      "evidence": "the specific message text or observation that characterizes this step",
      "confidence": 0.0,
      "msg_start": <absolute start index>,
      "msg_end": <absolute end index, exclusive>
    }}
  ]
}}

Identify between 1 and {max_nodes} reasoning steps. Prefer fewer, higher-quality \
nodes over many trivial ones. CONTEXT_LOAD is fine but don't fill the response with it."""

_EDGE_SYSTEM = """\
You are an expert at understanding how reasoning steps connect in AI agent sessions.
"""

_EDGE_PROMPT = """\
Given these reasoning nodes extracted from a coding agent session, identify which \
nodes led to which. Return ONLY the edges that are clearly supported by the node summaries.

Nodes:
{nodes_json}

For each edge, the relationship should be one of: \
"led_to", "contradicted", "refined", "resolved", "discovered_from", "caused_pivot_to"

Respond with JSON only:
{{
  "edges": [
    {{"source_id": "...", "target_id": "...", "relationship": "..."}}
  ]
}}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_message(msg: SessionMessage, index: int) -> str:
    """Format a single message for inclusion in a prompt."""
    role_label = msg.role.value.upper()
    content = msg.content

    if msg.role == MessageRole.TOOL_CALL:
        tool = msg.tool_name or "unknown"
        try:
            inp = json.loads(content)
            for k in ("content", "command"):
                if k in inp and isinstance(inp[k], str) and len(inp[k]) > 200:
                    inp[k] = inp[k][:200] + "..."
            content = f"[{tool}] {json.dumps(inp)}"
        except Exception:
            content = f"[{tool}] {content[:200]}"

    elif msg.role == MessageRole.TOOL_RESULT:
        content = content[:_MAX_TOOL_RESULT_CHARS]
        if len(msg.content) > _MAX_TOOL_RESULT_CHARS:
            content += "... [truncated]"

    return f"[{index}] {role_label}: {content}"


def _prefilter(messages: list[SessionMessage]) -> list[tuple[int, SessionMessage]]:
    """Remove obvious noise. Returns (original_index, message) pairs.

    Preserves the original message index so the LLM can reference absolute
    positions even after filtering.
    """
    filtered: list[tuple[int, SessionMessage]] = []
    for i, msg in enumerate(messages):
        if not msg.content.strip():
            continue
        if msg.role == MessageRole.TOOL_RESULT and len(msg.content) > _NOISE_TRUNCATE_THRESHOLD:
            short = SessionMessage(
                role=msg.role,
                content=msg.content[:_MAX_TOOL_RESULT_CHARS] + "... [truncated]",
                timestamp=msg.timestamp,
                tool_name=msg.tool_name,
            )
            filtered.append((i, short))
        else:
            filtered.append((i, msg))
    return filtered


# ── Token-budget windowing ───────────────────────────────────────────────────

@dataclass
class TokenWindow:
    """A contiguous range of messages that fits within a token budget."""
    start_msg_idx: int            # absolute index of first message in original session
    end_msg_idx: int              # absolute index of last message + 1
    messages: list[tuple[int, SessionMessage]]   # (abs_idx, msg) pairs
    token_count: int


class TokenBudgetWindower:
    """Pack session messages into windows that fit within a token budget.

    Uses Anthropic's count_tokens API to measure each message exactly. Builds
    windows greedily: keep adding messages until the next one would exceed
    the budget, then start a new window with token-based trailing overlap.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str = _MODEL,
        fill_ratio: float | None = None,
        output_reserve: int = DEFAULT_OUTPUT_RESERVE,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        prompt_overhead_tokens: int = 0,
    ):
        self.client = client
        self.model = model
        self.fill_ratio = fill_ratio if fill_ratio is not None else _resolve_fill_ratio()
        self.output_reserve = output_reserve
        self.overlap_tokens = overlap_tokens
        self.prompt_overhead_tokens = prompt_overhead_tokens

    @property
    def budget(self) -> int:
        """Effective token budget for raw message content per window."""
        b = int(MODEL_CONTEXT_WINDOW * self.fill_ratio)
        b -= self.output_reserve
        b -= self.prompt_overhead_tokens
        return max(MIN_WINDOW_TOKENS, b)

    async def count_tokens(self, text: str) -> int:
        """Count tokens for a single piece of text via Anthropic SDK."""
        try:
            resp = await self.client.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
            return int(resp.input_tokens)
        except Exception:
            # Fallback: rough estimate of 4 chars per token
            return max(1, len(text) // 4)

    async def count_messages(
        self, items: list[tuple[int, SessionMessage]]
    ) -> list[int]:
        """Concurrently count tokens for every message."""
        formatted = [_format_message(m, idx) for idx, m in items]
        tasks = [self.count_tokens(t) for t in formatted]
        return await asyncio.gather(*tasks)

    def pack(
        self,
        items: list[tuple[int, SessionMessage]],
        token_counts: list[int],
    ) -> list[TokenWindow]:
        """Pack messages into windows greedily within the token budget.

        Args:
            items: (absolute_index, message) pairs after noise filtering
            token_counts: parallel list of token counts for each item

        Returns:
            List of TokenWindow objects with token-based overlap between them.
        """
        if not items:
            return []

        budget = self.budget
        windows: list[TokenWindow] = []
        i = 0
        n = len(items)

        while i < n:
            # Greedy fill: pack messages until adding the next one breaks budget
            window_items: list[tuple[int, SessionMessage]] = []
            window_tokens = 0
            j = i
            while j < n:
                next_count = token_counts[j]
                if window_tokens + next_count > budget and window_items:
                    break
                window_items.append(items[j])
                window_tokens += next_count
                j += 1

            # Edge case: a single message exceeds budget — include it anyway
            if not window_items and i < n:
                window_items.append(items[i])
                window_tokens = token_counts[i]
                j = i + 1

            windows.append(TokenWindow(
                start_msg_idx=window_items[0][0],
                end_msg_idx=window_items[-1][0] + 1,
                messages=window_items,
                token_count=window_tokens,
            ))

            if j >= n:
                break

            # Compute trailing overlap: walk backward from j until we accumulate
            # ~overlap_tokens worth of messages, those become the start of the
            # next window.
            overlap_start = j
            overlap_acc = 0
            k = j - 1
            while k > i and overlap_acc < self.overlap_tokens:
                overlap_acc += token_counts[k]
                overlap_start = k
                k -= 1

            # Guard against zero-progress overlap
            if overlap_start <= i:
                overlap_start = j

            i = overlap_start

        return windows


def _resolve_fill_ratio() -> float:
    """Resolve the context fill ratio from env var, then config, then default."""
    env_val = os.environ.get("CMM_CONTEXT_FILL_RATIO")
    if env_val:
        try:
            v = float(env_val)
            if 0.05 <= v <= 0.95:
                return v
        except ValueError:
            pass
    return DEFAULT_FILL_RATIO


# ── LLM classification ──────────────────────────────────────────────────────

async def _classify_window(
    client: anthropic.AsyncAnthropic,
    window_idx: int,
    window: TokenWindow,
) -> list[ReasoningNode]:
    """Classify a single window via LLM. Returns a LIST of nodes (not just one).

    On failure, returns a single CONTEXT_LOAD fallback node spanning the window.
    """
    segment = "\n".join(_format_message(m, idx) for idx, m in window.messages)
    prompt = _CLASSIFICATION_PROMPT.format(
        segment=segment,
        max_nodes=_MAX_NODES_PER_WINDOW,
    )

    try:
        response = await client.messages.create(
            model=_CLASSIFICATION_MODEL,
            max_tokens=DEFAULT_OUTPUT_RESERVE,
            system=_CLASSIFICATION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data: dict[str, Any] = json.loads(raw)
        node_dicts = data.get("nodes", [])

        nodes: list[ReasoningNode] = []
        for i, nd in enumerate(node_dicts):
            try:
                node_type_str = str(nd.get("node_type", "CONTEXT_LOAD")).upper()
                try:
                    node_type = NodeType(node_type_str.lower())
                except ValueError:
                    node_type = NodeType.CONTEXT_LOAD

                msg_start = int(nd.get("msg_start", window.start_msg_idx))
                msg_end = int(nd.get("msg_end", window.end_msg_idx))
                # Clamp to window bounds for safety
                msg_start = max(window.start_msg_idx, min(msg_start, window.end_msg_idx))
                msg_end = max(msg_start + 1, min(msg_end, window.end_msg_idx))

                nodes.append(ReasoningNode(
                    node_id=f"node-{window_idx:03d}-{i:02d}",
                    node_type=node_type,
                    summary=str(nd.get("summary", "")),
                    evidence=str(nd.get("evidence", "")),
                    message_range=(msg_start, msg_end),
                    confidence=float(nd.get("confidence", 0.5)),
                ))
            except Exception:
                continue

        if not nodes:
            # LLM returned empty list — emit a fallback so the window isn't lost
            nodes.append(ReasoningNode(
                node_id=f"node-{window_idx:03d}-00",
                node_type=NodeType.CONTEXT_LOAD,
                summary=f"Window {window_idx}: no reasoning steps identified",
                evidence="",
                message_range=(window.start_msg_idx, window.end_msg_idx),
                confidence=0.2,
            ))
        return nodes

    except Exception as e:
        return [ReasoningNode(
            node_id=f"node-{window_idx:03d}-00",
            node_type=NodeType.CONTEXT_LOAD,
            summary=f"Window {window_idx}: extraction failed ({type(e).__name__})",
            evidence="",
            message_range=(window.start_msg_idx, window.end_msg_idx),
            confidence=0.0,
        )]


async def _build_edges(
    client: anthropic.AsyncAnthropic, nodes: list[ReasoningNode]
) -> list[ReasoningEdge]:
    """Ask the LLM to identify edges between classified nodes."""
    if len(nodes) < 2:
        return []

    nodes_json = json.dumps(
        [
            {
                "node_id": n.node_id,
                "node_type": n.node_type.value,
                "summary": n.summary,
            }
            for n in nodes
        ],
        indent=2,
    )
    prompt = _EDGE_PROMPT.format(nodes_json=nodes_json)

    try:
        response = await client.messages.create(
            model=_CLASSIFICATION_MODEL,
            max_tokens=1024,
            system=_EDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)

        edges = []
        node_ids = {n.node_id for n in nodes}
        for e in data.get("edges", []):
            src = e.get("source_id", "")
            tgt = e.get("target_id", "")
            rel = e.get("relationship", "led_to")
            if src in node_ids and tgt in node_ids and src != tgt:
                edges.append(ReasoningEdge(source_id=src, target_id=tgt, relationship=rel))
        return edges

    except Exception:
        return [
            ReasoningEdge(source_id=nodes[i].node_id, target_id=nodes[i + 1].node_id, relationship="led_to")
            for i in range(len(nodes) - 1)
        ]


def _detect_pivots(nodes: list[ReasoningNode], edges: list[ReasoningEdge]) -> list[str]:
    """Return node_ids of pivot or dead_end nodes."""
    pivot_types = {NodeType.PIVOT, NodeType.DEAD_END}
    return [n.node_id for n in nodes if n.node_type in pivot_types]


# ── Main class ───────────────────────────────────────────────────────────────

class DAGBuilder:
    """Extract a reasoning DAG from a normalized session using token-budget LLM analysis."""

    def __init__(
        self,
        api_key: str | None = None,
        fill_ratio: float | None = None,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        output_reserve: int = DEFAULT_OUTPUT_RESERVE,
        model: str = _MODEL,
    ):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.fill_ratio = fill_ratio if fill_ratio is not None else _resolve_fill_ratio()
        self.overlap_tokens = overlap_tokens
        self.output_reserve = output_reserve

        # Approximate prompt overhead — the classification prompt template is
        # roughly ~600 tokens of fixed scaffolding around the segment.
        self._prompt_overhead = 600

        self.windower = TokenBudgetWindower(
            client=self.client,
            model=self.model,
            fill_ratio=self.fill_ratio,
            output_reserve=self.output_reserve,
            overlap_tokens=self.overlap_tokens,
            prompt_overhead_tokens=self._prompt_overhead,
        )

    async def build(self, session: NormalizedSession) -> ReasoningDAG:
        original_count = len(session.messages)

        # Step 1: Filter noise (preserves absolute indices)
        filtered = _prefilter(session.messages)

        if not filtered:
            return ReasoningDAG(
                session_id=session.session_id,
                nodes=[],
                edges=[],
                pivot_nodes=[],
                noise_ratio=1.0 if original_count > 0 else 0.0,
            )

        # Step 2: Pre-count tokens for every message (concurrent)
        token_counts = await self.windower.count_messages(filtered)

        # Step 3: Pack into token-budget windows
        windows = self.windower.pack(filtered, token_counts)

        # Step 4: Classify each window concurrently — each may return MULTIPLE nodes
        tasks = [
            _classify_window(self.client, idx, window)
            for idx, window in enumerate(windows)
        ]
        nodes_per_window = await asyncio.gather(*tasks)

        # Flatten and dedupe by message_range overlap (windows have overlap, so
        # the same reasoning step may be classified twice).
        nodes = self._dedupe_overlapping_nodes(nodes_per_window)

        # Step 5: Build edges across all classified nodes
        edges = await _build_edges(self.client, nodes)

        # Step 6: Detect pivots
        pivots = _detect_pivots(nodes, edges)

        noise_ratio = 1.0 - (len(filtered) / original_count) if original_count > 0 else 0.0

        return ReasoningDAG(
            session_id=session.session_id,
            nodes=nodes,
            edges=edges,
            pivot_nodes=pivots,
            noise_ratio=noise_ratio,
        )

    @staticmethod
    def _dedupe_overlapping_nodes(
        nodes_per_window: list[list[ReasoningNode]],
    ) -> list[ReasoningNode]:
        """Merge nodes from overlapping windows by message_range proximity.

        Two nodes are considered duplicates if their message ranges overlap
        AND they share the same node_type. The higher-confidence one wins.
        """
        all_nodes: list[ReasoningNode] = []
        for batch in nodes_per_window:
            all_nodes.extend(batch)

        if not all_nodes:
            return []

        # Sort by start position, then descending confidence so the best version
        # of any duplicate is encountered first.
        all_nodes.sort(key=lambda n: (n.message_range[0], -n.confidence))

        kept: list[ReasoningNode] = []
        for node in all_nodes:
            is_dup = False
            for k in kept:
                if k.node_type != node.node_type:
                    continue
                # Overlap test: ranges intersect
                if (node.message_range[0] < k.message_range[1] and
                        node.message_range[1] > k.message_range[0]):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(node)

        # Re-sort by start position for downstream consumers
        kept.sort(key=lambda n: n.message_range[0])
        return kept
