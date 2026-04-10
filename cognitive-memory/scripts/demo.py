#!/usr/bin/env python3
"""End-to-end demo — run the full CMM pipeline on included fixtures.

This script is designed for someone who just cloned the repo and wants to
see the system work in under 2 minutes. It:

    1. Parses all 10 included session fixtures
    2. Extracts reasoning DAGs (warm tier — fast, no Anthropic key needed)
    3. Stores nodes in a local ChromaDB (needs OPENAI_API_KEY for embeddings)
    4. Builds a cognitive profile (if ANTHROPIC_API_KEY available)
    5. Runs a sample search query
    6. Generates an interactive HTML DAG visualization
    7. Prints a summary of everything it did

Usage:
    # Full demo (needs OPENAI_API_KEY)
    python scripts/demo.py

    # Parse-only mode (no API keys needed — skips storage and search)
    python scripts/demo.py --parse-only

    # Specify a different output directory
    python scripts/demo.py --output output/demo/

    # Clean up after demo
    python scripts/demo.py --clean
"""
import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()

_PROJECT_ROOT = Path(__file__).parent.parent
_FIXTURES_DIR = _PROJECT_ROOT / "fixtures"
_DEMO_STORE = _PROJECT_ROOT / "data" / "demo_store"
_DEMO_PROJECT = "demo"
_DEMO_OUTPUT = _PROJECT_ROOT / "output" / "demo"


def _find_fixtures() -> list[Path]:
    """Find all .jsonl fixture files."""
    fixtures = []
    for subdir in ("claude_code", "synthetic"):
        d = _FIXTURES_DIR / subdir
        if d.exists():
            fixtures.extend(sorted(d.glob("*.jsonl")))
    return fixtures


def _clean(store_dir: Path, output_dir: Path):
    """Remove demo artifacts."""
    for d in (store_dir, output_dir):
        if d.exists():
            shutil.rmtree(d)
            console.print(f"  Removed {d}")


def run_demo(parse_only: bool = False, output_dir: Path = _DEMO_OUTPUT, store_dir: Path = _DEMO_STORE):
    console.print(Panel(
        "[bold]Cognitive Memory Manager — Demo[/bold]\n"
        "Running the full pipeline on included fixture sessions.",
        style="blue",
    ))
    console.print()

    fixtures = _find_fixtures()
    if not fixtures:
        console.print("[red]No fixture files found in fixtures/[/red]")
        return

    console.print(f"Found [cyan]{len(fixtures)}[/cyan] fixture sessions\n")

    # ── Phase 1: Parse ──────────────────────────────────────────────
    console.rule("[bold]Phase 1: Parse sessions")
    from src.ingestion import ClaudeCodeParser
    parser = ClaudeCodeParser()

    sessions = []
    parse_table = Table(show_header=True)
    parse_table.add_column("File", style="cyan", max_width=35)
    parse_table.add_column("Messages", justify="right")
    parse_table.add_column("Files Referenced", justify="right")

    for fixture in fixtures:
        try:
            session = parser.parse_file(fixture)
            sessions.append((fixture, session))
            files_ref = sum(len(m.files_referenced) for m in session.messages)
            parse_table.add_row(
                fixture.name,
                str(len(session.messages)),
                str(files_ref),
            )
        except Exception as e:
            parse_table.add_row(fixture.name, "[red]failed[/red]", str(e)[:30])

    console.print(parse_table)
    total_messages = sum(len(s.messages) for _, s in sessions)
    console.print(f"\n  Parsed [green]{len(sessions)}[/green] sessions, "
                  f"[green]{total_messages}[/green] total messages\n")

    # ── Phase 2: Extract DAGs (warm tier) ───────────────────────────
    console.rule("[bold]Phase 2: Extract reasoning DAGs (warm tier)")
    from src.extraction.warm_extractor import WarmExtractor
    extractor = WarmExtractor()

    dags = []
    extract_table = Table(show_header=True)
    extract_table.add_column("Session", style="cyan", max_width=35)
    extract_table.add_column("Nodes", justify="right")
    extract_table.add_column("Pivots", justify="right")
    extract_table.add_column("Noise %", justify="right")

    for fixture, session in sessions:
        dag = extractor.extract(session)
        dags.append((fixture, session, dag))
        extract_table.add_row(
            fixture.name,
            str(len(dag.nodes)),
            str(len(dag.pivot_nodes)),
            f"{dag.noise_ratio:.0%}",
        )

    console.print(extract_table)
    total_nodes = sum(len(d.nodes) for _, _, d in dags)
    total_pivots = sum(len(d.pivot_nodes) for _, _, d in dags)
    console.print(f"\n  Extracted [green]{total_nodes}[/green] reasoning nodes, "
                  f"[yellow]{total_pivots}[/yellow] pivots\n")

    if parse_only:
        console.print("[yellow]--parse-only mode: skipping storage, search, and visualization.[/yellow]")
        console.print("[dim]Re-run without --parse-only to see the full demo (needs OPENAI_API_KEY).[/dim]")
        return

    # ── Phase 3: Store + Deduplicate ────────────────────────────────
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if not has_openai:
        console.print("[yellow]OPENAI_API_KEY not set — skipping storage, search, and profile.[/yellow]")
        console.print("[dim]Set OPENAI_API_KEY and re-run to see the full demo.[/dim]")
        console.print()
        _print_parse_only_summary(sessions, dags)
        return

    console.rule("[bold]Phase 3: Store + deduplicate")
    from src.store.vector_store import MemoryStore
    from src.compression.dedup import SemanticDeduplicator

    store = MemoryStore(persist_dir=str(store_dir))
    dedup = SemanticDeduplicator(store)

    stored_total = 0
    merged_total = 0
    dropped_total = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Storing nodes...", total=len(dags))
        for fixture, session, dag in dags:
            result = dedup.deduplicate(dag.nodes, _DEMO_PROJECT, session.session_id)
            from src.schemas.reasoning import ReasoningDAG
            dag_to_store = ReasoningDAG(
                session_id=dag.session_id,
                nodes=result.stored,
                edges=dag.edges,
                pivot_nodes=dag.pivot_nodes,
                noise_ratio=dag.noise_ratio,
            )
            count = store.store_dag(dag_to_store, _DEMO_PROJECT)
            stored_total += count
            merged_total += len(result.merged)
            dropped_total += len(result.dropped)
            progress.advance(task)

    console.print(f"\n  Stored [green]{stored_total}[/green] nodes, "
                  f"merged [yellow]{merged_total}[/yellow], "
                  f"dropped [dim]{dropped_total}[/dim]\n")

    # ── Phase 4: Build profile (if Anthropic key available) ─────────
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_anthropic:
        console.rule("[bold]Phase 4: Build cognitive profile")
        from src.compression.profile_builder import ProfileBuilder

        builder = ProfileBuilder()
        profile = asyncio.run(builder.build_profile(_DEMO_PROJECT, store))

        console.print(f"  Architectural insights: [green]{len(profile.architectural_insights)}[/green]")
        console.print(f"  Known pitfalls:         [yellow]{len(profile.pitfalls)}[/yellow]")
        console.print(f"  Diagnostic strategies:  [cyan]{len(profile.diagnostic_strategies)}[/cyan]")
        console.print(f"  Key patterns:           {len(profile.key_patterns)}")
        console.print(f"  Anti-patterns:          {len(profile.anti_patterns)}")
        console.print()

        if profile.pitfalls:
            console.print("[bold]Top pitfalls:[/bold]")
            for p in profile.pitfalls[:3]:
                console.print(f"  [{p.severity.upper()}] {p.description}")
            console.print()
    else:
        console.print("[dim]ANTHROPIC_API_KEY not set — skipping profile building.[/dim]")
        console.print("[dim]You can build one later: cmm consolidate -p demo[/dim]\n")

    # ── Phase 5: Sample search ──────────────────────────────────────
    console.rule("[bold]Phase 5: Search memory")
    query = "debugging test failures"
    results = store.search(query, project_id=_DEMO_PROJECT, top_k=5)
    console.print(f"  Query: [cyan]\"{query}\"[/cyan]")
    console.print(f"  Results: {len(results)}\n")

    if results:
        search_table = Table(show_header=True)
        search_table.add_column("#", style="dim", width=3)
        search_table.add_column("Type", width=14)
        search_table.add_column("Summary", max_width=60)
        search_table.add_column("Sim", justify="right", width=6)

        for i, r in enumerate(results, 1):
            ntype = r.get("node_type", "?").upper()
            sim = r.get("similarity", 0)
            summary = r.get("summary", "")[:60]
            search_table.add_row(str(i), ntype, summary, f"{sim:.2f}")

        console.print(search_table)
        console.print()

    # ── Phase 6: Generate visualization ─────────────────────────────
    console.rule("[bold]Phase 6: Generate DAG visualization")
    try:
        from scripts.visualize_dag import load_from_chromadb, generate_html

        dag_data = load_from_chromadb(str(store_dir), _DEMO_PROJECT)
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "demo_dag.html"
        generate_html(dag_data, str(html_path))
        console.print(f"  [green]Visualization:[/green] {html_path}")
        console.print(f"  Open in browser: [cyan]open {html_path}[/cyan]\n")
    except Exception as e:
        console.print(f"  [dim]Visualization skipped: {e}[/dim]\n")

    # ── Summary ─────────────────────────────────────────────────────
    console.rule("[bold green]Demo complete")
    console.print()

    summary_table = Table(title="Demo Summary", show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value")
    summary_table.add_row("Sessions parsed", str(len(sessions)))
    summary_table.add_row("Total messages", str(total_messages))
    summary_table.add_row("Reasoning nodes extracted", str(total_nodes))
    summary_table.add_row("Nodes stored (after dedup)", str(stored_total))
    summary_table.add_row("Store location", str(store_dir))
    summary_table.add_row("Project ID", _DEMO_PROJECT)
    console.print(summary_table)

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  # Search your new memory store")
    console.print(f"  cmm status --project {_DEMO_PROJECT} --store-dir {store_dir}")
    console.print()
    console.print("  # Build a profile (needs ANTHROPIC_API_KEY)")
    console.print(f"  cmm consolidate --project {_DEMO_PROJECT} --store-dir {store_dir}")
    console.print()
    console.print("  # Initialize your own project")
    console.print("  cd /path/to/your/project && cmm init .")
    console.print()
    console.print("  # Clean up demo artifacts")
    console.print("  python scripts/demo.py --clean")


def _print_parse_only_summary(sessions, dags):
    """Print summary when we can't store."""
    console.rule("[bold green]Demo complete (parse + extract only)")
    console.print()
    total_messages = sum(len(s.messages) for _, s in sessions)
    total_nodes = sum(len(d.nodes) for _, _, d in dags)

    # Show the most interesting nodes across all DAGs
    from src.schemas.reasoning import NodeType
    interesting_types = {NodeType.DISCOVERY, NodeType.PIVOT, NodeType.DEAD_END, NodeType.SOLUTION}
    interesting = []
    for _, _, dag in dags:
        for n in dag.nodes:
            if n.node_type in interesting_types:
                interesting.append(n)

    if interesting:
        interesting.sort(key=lambda n: -n.confidence)
        console.print("[bold]Most interesting extracted nodes:[/bold]\n")
        for n in interesting[:8]:
            ntype = n.node_type.value.upper()
            console.print(f"  [{ntype}] (conf={n.confidence:.2f}) {n.summary[:80]}")
        console.print()

    console.print(f"  Parsed {len(sessions)} sessions ({total_messages} messages)")
    console.print(f"  Extracted {total_nodes} reasoning nodes")
    console.print()
    console.print("[bold]To see the full demo:[/bold]")
    console.print("  export OPENAI_API_KEY='sk-...'")
    console.print("  python scripts/demo.py")


def main():
    parser = argparse.ArgumentParser(description="CMM end-to-end demo")
    parser.add_argument("--parse-only", action="store_true",
                        help="Parse and extract only (no API keys needed)")
    parser.add_argument("--output", default=str(_DEMO_OUTPUT),
                        help="Output directory for visualizations")
    parser.add_argument("--store-dir", default=str(_DEMO_STORE),
                        help="ChromaDB store directory")
    parser.add_argument("--clean", action="store_true",
                        help="Remove demo artifacts and exit")
    args = parser.parse_args()

    if args.clean:
        console.print("[bold]Cleaning demo artifacts...[/bold]")
        _clean(Path(args.store_dir), Path(args.output))
        console.print("[green]Done.[/green]")
        return

    run_demo(
        parse_only=args.parse_only,
        output_dir=Path(args.output),
        store_dir=Path(args.store_dir),
    )


if __name__ == "__main__":
    main()
