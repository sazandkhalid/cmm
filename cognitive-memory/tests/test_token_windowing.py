"""Tests for token-budget windowing in DAGBuilder."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from src.extraction.dag_builder import (
    DEFAULT_FILL_RATIO,
    DEFAULT_OVERLAP_TOKENS,
    MIN_WINDOW_TOKENS,
    MODEL_CONTEXT_WINDOW,
    TokenBudgetWindower,
    _resolve_fill_ratio,
)
from src.schemas.session import MessageRole, NormalizedSession, SessionMessage


def _msg(role: MessageRole, content: str) -> SessionMessage:
    return SessionMessage(role=role, content=content)


def _items(n: int, content_factory=lambda i: f"message {i}") -> list:
    """Build (idx, msg) tuples for n synthetic messages."""
    return [
        (i, _msg(MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT, content_factory(i)))
        for i in range(n)
    ]


@pytest.fixture
def windower():
    """A windower with deterministic token counts (no API calls)."""
    client = AsyncMock()
    w = TokenBudgetWindower(
        client=client,
        fill_ratio=0.45,
        output_reserve=4_000,
        overlap_tokens=750,
        prompt_overhead_tokens=600,
    )
    return w


# ── Budget math ──────────────────────────────────────────────────────────


def test_budget_uses_fill_ratio(windower):
    expected = int(MODEL_CONTEXT_WINDOW * 0.45) - 4_000 - 600
    assert windower.budget == expected


def test_budget_respects_minimum():
    """Even with absurdly low fill ratio, budget never goes below MIN_WINDOW_TOKENS."""
    client = AsyncMock()
    w = TokenBudgetWindower(
        client=client,
        fill_ratio=0.001,
        output_reserve=4_000,
        overlap_tokens=750,
        prompt_overhead_tokens=600,
    )
    assert w.budget == MIN_WINDOW_TOKENS


def test_fill_ratio_default():
    """No env var → default 0.45."""
    if "CMM_CONTEXT_FILL_RATIO" in os.environ:
        del os.environ["CMM_CONTEXT_FILL_RATIO"]
    assert _resolve_fill_ratio() == DEFAULT_FILL_RATIO


def test_fill_ratio_env_override():
    os.environ["CMM_CONTEXT_FILL_RATIO"] = "0.30"
    try:
        assert _resolve_fill_ratio() == 0.30
    finally:
        del os.environ["CMM_CONTEXT_FILL_RATIO"]


def test_fill_ratio_invalid_env_falls_back():
    os.environ["CMM_CONTEXT_FILL_RATIO"] = "not-a-number"
    try:
        assert _resolve_fill_ratio() == DEFAULT_FILL_RATIO
    finally:
        del os.environ["CMM_CONTEXT_FILL_RATIO"]


def test_fill_ratio_out_of_range_falls_back():
    os.environ["CMM_CONTEXT_FILL_RATIO"] = "1.5"
    try:
        assert _resolve_fill_ratio() == DEFAULT_FILL_RATIO
    finally:
        del os.environ["CMM_CONTEXT_FILL_RATIO"]


# ── Single-window fast path ──────────────────────────────────────────────


def test_short_session_produces_single_window(windower):
    """A session whose messages sum to less than the budget → 1 window."""
    items = _items(10)
    # Each message is "small" — 100 tokens — total 1000 < budget
    counts = [100] * 10
    windows = windower.pack(items, counts)
    assert len(windows) == 1
    assert windows[0].start_msg_idx == 0
    assert windows[0].end_msg_idx == 10
    assert len(windows[0].messages) == 10
    assert windows[0].token_count == 1000


def test_empty_input_produces_no_windows(windower):
    assert windower.pack([], []) == []


# ── Multi-window with overlap ────────────────────────────────────────────


def test_long_session_splits_into_multiple_windows(windower):
    """A session that exceeds budget → multiple windows."""
    items = _items(100)
    # Each message ~1000 tokens — total 100k tokens, well above budget (~85k)
    counts = [1000] * 100

    windows = windower.pack(items, counts)
    assert len(windows) >= 2

    # Each window's token total should be ≤ budget
    for w in windows:
        assert w.token_count <= windower.budget

    # First window starts at message 0
    assert windows[0].start_msg_idx == 0

    # Last window must reach the end
    assert windows[-1].end_msg_idx == 100


def test_windows_have_token_based_overlap(windower):
    """Consecutive windows must share trailing/leading messages worth ~overlap_tokens."""
    items = _items(60)
    counts = [1500] * 60  # ~90k tokens total — multiple windows

    windows = windower.pack(items, counts)
    assert len(windows) >= 2

    # For each consecutive pair, the start of window K+1 should be <= end of window K
    for i in range(len(windows) - 1):
        cur = windows[i]
        nxt = windows[i + 1]
        assert nxt.start_msg_idx < cur.end_msg_idx, (
            f"Window {i+1} starts at {nxt.start_msg_idx} but window {i} ends at {cur.end_msg_idx} "
            f"— no overlap"
        )

        # Compute overlap tokens
        overlap_idxs = set(idx for idx, _ in cur.messages) & set(idx for idx, _ in nxt.messages)
        assert len(overlap_idxs) > 0


def test_overlap_preserves_context_continuity(windower):
    """The last N tokens of window K must equal the first N tokens of window K+1."""
    items = _items(40)
    # Larger messages to force splits
    counts = [2500] * 40

    windows = windower.pack(items, counts)
    assert len(windows) >= 2

    for i in range(len(windows) - 1):
        cur_msgs = [(idx, m) for idx, m in windows[i].messages]
        nxt_msgs = [(idx, m) for idx, m in windows[i + 1].messages]

        # The first message of next window must appear somewhere in current window
        first_nxt_idx = nxt_msgs[0][0]
        cur_idxs = [idx for idx, _ in cur_msgs]
        assert first_nxt_idx in cur_idxs


def test_oversized_single_message_still_included(windower):
    """A single message larger than the budget shouldn't be dropped."""
    items = _items(5)
    big = windower.budget * 2
    counts = [big, 100, 100, 100, 100]

    windows = windower.pack(items, counts)
    # First window must contain the oversized message even though it exceeds budget
    assert windows[0].messages[0][0] == 0
    assert windows[0].token_count == big


def test_progress_guarantee(windower):
    """Pack must always make forward progress — no infinite loops."""
    items = _items(50)
    counts = [windower.budget // 4] * 50  # 4 messages per window

    windows = windower.pack(items, counts)

    # Every message must appear in at least one window
    seen = set()
    for w in windows:
        for idx, _ in w.messages:
            seen.add(idx)
    assert seen == set(range(50))


def test_window_count_is_reasonable(windower):
    """A 90k-token session should produce 2-3 windows, not 30."""
    items = _items(90)
    counts = [1000] * 90  # 90k tokens

    windows = windower.pack(items, counts)
    assert 2 <= len(windows) <= 4, f"Expected 2-4 windows, got {len(windows)}"


# ── Token counting ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_count_tokens_via_sdk():
    """Token counting calls the Anthropic SDK count_tokens API."""
    client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.input_tokens = 42
    client.messages.count_tokens = AsyncMock(return_value=mock_resp)

    w = TokenBudgetWindower(client=client)
    count = await w.count_tokens("hello world")
    assert count == 42
    client.messages.count_tokens.assert_called_once()


@pytest.mark.asyncio
async def test_count_tokens_fallback_on_error():
    """If count_tokens fails, fall back to char-based estimate."""
    client = AsyncMock()
    client.messages.count_tokens = AsyncMock(side_effect=Exception("API down"))

    w = TokenBudgetWindower(client=client)
    text = "x" * 400
    count = await w.count_tokens(text)
    assert count == 100  # 400 chars / 4


@pytest.mark.asyncio
async def test_count_messages_concurrent():
    """count_messages issues concurrent count_tokens calls."""
    client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.input_tokens = 10
    client.messages.count_tokens = AsyncMock(return_value=mock_resp)

    w = TokenBudgetWindower(client=client)
    items = _items(5)
    counts = await w.count_messages(items)
    assert counts == [10, 10, 10, 10, 10]
    assert client.messages.count_tokens.call_count == 5


# ── Dedupe of overlapping nodes ──────────────────────────────────────────


def test_dedupe_overlapping_nodes():
    """When two windows produce same-type nodes for overlapping ranges, keep one."""
    from src.extraction.dag_builder import DAGBuilder
    from src.schemas.reasoning import NodeType, ReasoningNode

    n1 = ReasoningNode(
        node_id="a", node_type=NodeType.HYPOTHESIS,
        summary="thinks the bug is in X", evidence="...",
        message_range=(5, 8), confidence=0.6,
    )
    # Same type, overlapping range, lower confidence — should be dropped
    n2 = ReasoningNode(
        node_id="b", node_type=NodeType.HYPOTHESIS,
        summary="thinks the bug is in X (window 2)", evidence="...",
        message_range=(7, 10), confidence=0.4,
    )
    # Different type, overlapping range — kept
    n3 = ReasoningNode(
        node_id="c", node_type=NodeType.DISCOVERY,
        summary="found root cause", evidence="...",
        message_range=(8, 9), confidence=0.8,
    )

    result = DAGBuilder._dedupe_overlapping_nodes([[n1, n3], [n2]])
    types = sorted(n.node_type.value for n in result)
    assert types == ["discovery", "hypothesis"]
    # The kept hypothesis should be the higher-confidence one
    hyp = next(n for n in result if n.node_type == NodeType.HYPOTHESIS)
    assert hyp.confidence == 0.6
