#!/usr/bin/env python3
"""Explore the ChromaDB memory store -- see what's inside.

Usage:
    uv run python cognitive-memory/scripts/explore_store.py
    uv run python cognitive-memory/scripts/explore_store.py --store-dir /path/to/store
    uv run python cognitive-memory/scripts/explore_store.py --project mcp-gateway-registry
    uv run python cognitive-memory/scripts/explore_store.py --node-id node-000-01
    uv run python cognitive-memory/scripts/explore_store.py --search "debugging pydantic"
    uv run python cognitive-memory/scripts/explore_store.py --type pivot
    uv run python cognitive-memory/scripts/explore_store.py --limit 20
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

console = Console()

_DEFAULT_STORE = Path(__file__).parent.parent / "data" / "memory_store"

_COLLECTION_NODES = "reasoning_nodes"
_COLLECTION_PROFILES = "cognitive_profiles"


def _get_client(
    store_dir: str,
) -> chromadb.ClientAPI:
    """Create a read-only ChromaDB client."""
    return chromadb.PersistentClient(
        path=store_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _show_overview(
    client: chromadb.ClientAPI,
) -> None:
    """Show high-level overview of all collections."""
    collections = client.list_collections()
    console.rule("[bold]Store Overview")

    table = Table(show_header=True)
    table.add_column("Collection", style="cyan")
    table.add_column("Count", justify="right")

    for col in collections:
        table.add_row(col.name, str(col.count()))

    console.print(table)


def _show_projects(
    nodes_col: chromadb.Collection,
) -> None:
    """Show breakdown by project."""
    count = nodes_col.count()
    if count == 0:
        console.print("[yellow]No nodes in store.[/yellow]")
        return

    results = nodes_col.get(include=["metadatas"])
    metas = results["metadatas"] or []

    project_counts: Counter = Counter()
    for m in metas:
        project_counts[m.get("project_id", "unknown")] += 1

    console.rule("[bold]Projects")
    table = Table(show_header=True)
    table.add_column("Project", style="cyan")
    table.add_column("Nodes", justify="right")

    for project, cnt in project_counts.most_common():
        table.add_row(project, str(cnt))

    console.print(table)


def _show_node_types(
    nodes_col: chromadb.Collection,
    project_id: str | None = None,
) -> None:
    """Show breakdown by node type."""
    count = nodes_col.count()
    if count == 0:
        return

    results = nodes_col.get(include=["metadatas"])
    metas = results["metadatas"] or []

    if project_id:
        metas = [m for m in metas if m.get("project_id") == project_id]

    type_counts: Counter = Counter()
    confidence_sums: dict[str, float] = {}
    confidence_counts: dict[str, int] = {}

    for m in metas:
        ntype = m.get("node_type", "unknown")
        conf = float(m.get("confidence", 0))
        type_counts[ntype] += 1
        confidence_sums[ntype] = confidence_sums.get(ntype, 0) + conf
        confidence_counts[ntype] = confidence_counts.get(ntype, 0) + 1

    title = "Node Types"
    if project_id:
        title += f" (project: {project_id})"
    console.rule(f"[bold]{title}")

    table = Table(show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Avg Confidence", justify="right")

    for ntype, cnt in type_counts.most_common():
        avg_conf = confidence_sums[ntype] / confidence_counts[ntype]
        table.add_row(ntype, str(cnt), f"{avg_conf:.2f}")

    console.print(table)


def _show_sessions(
    nodes_col: chromadb.Collection,
    project_id: str | None = None,
) -> None:
    """Show breakdown by session."""
    count = nodes_col.count()
    if count == 0:
        return

    results = nodes_col.get(include=["metadatas"])
    metas = results["metadatas"] or []

    if project_id:
        metas = [m for m in metas if m.get("project_id") == project_id]

    session_counts: Counter = Counter()
    for m in metas:
        session_counts[m.get("session_id", "unknown")] += 1

    title = "Sessions"
    if project_id:
        title += f" (project: {project_id})"
    console.rule(f"[bold]{title}")

    table = Table(show_header=True)
    table.add_column("Session ID", style="cyan", max_width=40)
    table.add_column("Nodes", justify="right")

    for session_id, cnt in session_counts.most_common(20):
        table.add_row(session_id[:40], str(cnt))

    if len(session_counts) > 20:
        console.print(f"  ... and {len(session_counts) - 20} more sessions")

    console.print(table)
    console.print(f"\nTotal sessions: [bold]{len(session_counts)}[/bold]")


def _show_nodes(
    nodes_col: chromadb.Collection,
    project_id: str | None = None,
    node_type: str | None = None,
    limit: int = 10,
) -> None:
    """Show individual nodes."""
    count = nodes_col.count()
    if count == 0:
        console.print("[yellow]No nodes in store.[/yellow]")
        return

    results = nodes_col.get(
        include=["metadatas", "documents"],
    )
    ids = results["ids"] or []
    metas = results["metadatas"] or []
    docs = results["documents"] or []

    # Filter
    filtered = []
    for i, m in enumerate(metas):
        if project_id and m.get("project_id") != project_id:
            continue
        if node_type and m.get("node_type", "").lower() != node_type.lower():
            continue
        filtered.append(i)

    title = f"Nodes (showing {min(limit, len(filtered))} of {len(filtered)})"
    console.rule(f"[bold]{title}")

    for idx in filtered[:limit]:
        m = metas[idx]
        doc = docs[idx] if idx < len(docs) else ""
        ntype = m.get("node_type", "?")
        conf = float(m.get("confidence", 0))
        session = m.get("session_id", "?")[:20]
        msg_range = f"[{m.get('msg_start', '?')}-{m.get('msg_end', '?')}]"

        # Color by type
        type_colors = {
            "hypothesis": "yellow",
            "investigation": "blue",
            "discovery": "green",
            "pivot": "magenta",
            "solution": "bold green",
            "dead_end": "red",
            "context_load": "dim",
        }
        color = type_colors.get(ntype.lower(), "white")

        console.print(Panel(
            f"[{color}]{ntype.upper()}[/{color}] (confidence: {conf:.2f}) "
            f"msgs {msg_range}  session: {session}\n\n"
            f"{doc or m.get('summary', 'no summary')}",
            title=f"[cyan]{ids[idx]}[/cyan]",
            width=100,
        ))


def _show_single_node(
    nodes_col: chromadb.Collection,
    node_id: str,
) -> None:
    """Show full details for a single node."""
    results = nodes_col.get(
        ids=[node_id],
        include=["metadatas", "documents", "embeddings"],
    )

    if not results["ids"]:
        console.print(f"[red]Node not found: {node_id}[/red]")
        return

    meta = results["metadatas"][0]
    doc = results["documents"][0] if results["documents"] else ""
    emb = results["embeddings"][0] if results["embeddings"] else None

    console.rule(f"[bold]Node: {node_id}")
    console.print(f"[bold]Metadata:[/bold]\n{json.dumps(meta, indent=2, default=str)}\n")
    console.print(f"[bold]Document:[/bold]\n{doc}\n")
    if emb:
        console.print(f"[bold]Embedding:[/bold] {len(emb)} dimensions, "
                       f"first 5: {emb[:5]}")


def _search_nodes(
    nodes_col: chromadb.Collection,
    query: str,
    store_dir: str,
    project_id: str | None = None,
    limit: int = 10,
) -> None:
    """Semantic search over nodes."""
    from src.store.vector_store import MemoryStore

    store = MemoryStore(persist_dir=store_dir)
    embeddings = store.embed([query])

    where_filter = {"project_id": project_id} if project_id else None

    results = nodes_col.query(
        query_embeddings=embeddings,
        n_results=min(limit, nodes_col.count() or 1),
        include=["metadatas", "documents", "distances"],
        where=where_filter,
    )

    console.rule(f"[bold]Search: \"{query}\"")

    if not results["ids"] or not results["ids"][0]:
        console.print("[yellow]No results.[/yellow]")
        return

    for i, node_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i] if results["documents"] else ""
        distance = results["distances"][0][i] if results["distances"] else 0
        similarity = 1.0 - distance

        ntype = meta.get("node_type", "?")
        conf = float(meta.get("confidence", 0))

        type_colors = {
            "hypothesis": "yellow",
            "investigation": "blue",
            "discovery": "green",
            "pivot": "magenta",
            "solution": "bold green",
            "dead_end": "red",
            "context_load": "dim",
        }
        color = type_colors.get(ntype.lower(), "white")

        console.print(Panel(
            f"[{color}]{ntype.upper()}[/{color}] "
            f"similarity: {similarity:.3f}  confidence: {conf:.2f}\n\n"
            f"{doc or meta.get('summary', 'no summary')}",
            title=f"[cyan]{node_id}[/cyan]",
            width=100,
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Explore the ChromaDB memory store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Overview of the store
    uv run python cognitive-memory/scripts/explore_store.py

    # Filter by project
    uv run python cognitive-memory/scripts/explore_store.py --project mcp-gateway-registry

    # Show only pivot nodes
    uv run python cognitive-memory/scripts/explore_store.py --type pivot

    # Semantic search
    uv run python cognitive-memory/scripts/explore_store.py --search "debugging pydantic"

    # Look up a specific node
    uv run python cognitive-memory/scripts/explore_store.py --node-id node-000-01
""",
    )
    parser.add_argument(
        "--store-dir",
        default=str(_DEFAULT_STORE),
        help=f"Memory store directory (default: {_DEFAULT_STORE})",
    )
    parser.add_argument(
        "--project", "-p",
        default=None,
        help="Filter by project ID",
    )
    parser.add_argument(
        "--type", "-t",
        default=None,
        help="Filter by node type (hypothesis, investigation, discovery, pivot, solution, dead_end)",
    )
    parser.add_argument(
        "--node-id",
        default=None,
        help="Show full details for a specific node ID",
    )
    parser.add_argument(
        "--search", "-s",
        default=None,
        help="Semantic search query",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Max nodes to display (default: 10)",
    )
    parser.add_argument(
        "--nodes-only",
        action="store_true",
        help="Skip overview, jump straight to node listing",
    )
    args = parser.parse_args()

    store_path = args.store_dir
    if not Path(store_path).exists():
        console.print(f"[red]Store not found: {store_path}[/red]")
        sys.exit(1)

    client = _get_client(store_path)
    nodes_col = client.get_or_create_collection(
        name=_COLLECTION_NODES,
        metadata={"hnsw:space": "cosine"},
    )

    # Single node lookup
    if args.node_id:
        _show_single_node(nodes_col, args.node_id)
        return

    # Semantic search
    if args.search:
        _search_nodes(nodes_col, args.search, store_path, args.project, args.limit)
        return

    # Overview
    if not args.nodes_only:
        _show_overview(client)
        _show_projects(nodes_col)
        _show_node_types(nodes_col, args.project)
        _show_sessions(nodes_col, args.project)

    # Node listing
    _show_nodes(nodes_col, args.project, args.type, args.limit)


if __name__ == "__main__":
    main()
