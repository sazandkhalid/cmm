"""cmm — cognitive memory CLI entry point."""
import click


@click.group()
def main():
    """Cognitive Memory Manager — persistent cross-platform memory for AI coding agents."""


# ── cmm init ──────────────────────────────────────────────────────────────────

@main.command()
@click.argument("target", default=".", type=click.Path(exists=True))
@click.option("--store-dir", default=None, help="Local ChromaDB store path")
@click.option("--shared", "shared_path", default=None, help="Shared team store path (filesystem shared mode)")
@click.option("--developer", default=None, help="Your name (for sync attribution)")
@click.option("--team-id", default=None, help="Team identifier")
@click.option("--cloud-tenant", default=None, envvar="CMM_CHROMA_TENANT",
              help="Chroma Cloud tenant ID (enables cloud shared mode)")
@click.option("--cloud-database", default=None, envvar="CMM_CHROMA_DATABASE",
              help="Chroma Cloud database name (default: cmm)")
def init(target, store_dir, shared_path, developer, team_id, cloud_tenant, cloud_database):
    """Initialize cognitive memory for a project.

    Creates the .cognitive/ folder. With --shared or --cloud-tenant/--cloud-database,
    registers a team store and immediately runs cmm pull to populate the local
    cache from existing approved team memories.

    Chroma Cloud mode (recommended for teams):

        cmm init . --cloud-tenant <tenant-id> --cloud-database cmm \\
                   --developer alice

        Set CMM_CHROMA_API_KEY in your environment before running.
    """
    from pathlib import Path
    import os
    from rich.console import Console

    from src.discovery.project import CognitiveProject
    from src.discovery.llms_txt import generate_llms_txt

    console = Console()
    project_dir = Path(target).resolve()

    # Determine shared mode
    _cloud_mode = bool(cloud_tenant and cloud_database)
    _shared_mode = _cloud_mode or bool(shared_path)

    cognitive_dir = project_dir / ".cognitive"
    if (cognitive_dir / "manifest.json").exists():
        proj = CognitiveProject.load(project_dir)
        console.print(f"[yellow]Already initialized:[/yellow] {proj.project_id}")
        console.print(f"  .cognitive/ exists at {cognitive_dir}")
        # If --shared/--cloud was passed on a re-init, still update the config
        if _shared_mode or developer or team_id:
            if shared_path:
                proj.config["shared_store_path"] = shared_path
                proj.config["mode"] = "shared"
            if _cloud_mode:
                proj.config["cloud_tenant"] = cloud_tenant
                proj.config["cloud_database"] = cloud_database
                proj.config["mode"] = "cloud"
            if developer:
                proj.config["developer_name"] = developer
            if team_id:
                proj.config["team_id"] = team_id
            proj.save_config()
            console.print(f"  [green]Updated config[/green]")
        return

    proj = CognitiveProject.init(project_dir, store_path=store_dir)

    # Stamp shared/cloud-mode config fields
    if shared_path:
        proj.config["shared_store_path"] = shared_path
        proj.config["mode"] = "shared"
    if _cloud_mode:
        proj.config["cloud_tenant"] = cloud_tenant
        proj.config["cloud_database"] = cloud_database
        proj.config["mode"] = "cloud"
        # Note: API key is NOT stored in config — use CMM_CHROMA_API_KEY env var
    if developer:
        proj.config["developer_name"] = developer
    if team_id:
        proj.config["team_id"] = team_id
    if store_dir:
        proj.config["local_store_path"] = store_dir
    proj.save_config()

    # Generate initial llms.txt
    llms_content = generate_llms_txt(
        project_name=proj.name,
        project_description=proj.description,
        profile=None,
        project_dir=project_dir,
    )
    proj.llms_txt_path.write_text(llms_content)

    console.print(f"[green]Initialized cognitive memory:[/green]")
    console.print(f"  Project ID:  [cyan]{proj.project_id}[/cyan]")
    console.print(f"  Name:        {proj.name}")
    console.print(f"  Description: {proj.description[:80] or '(none)'}")
    console.print(f"  Directory:   {cognitive_dir}")
    if _cloud_mode:
        console.print(f"  Mode:        [magenta]cloud (Chroma Cloud)[/magenta]")
        console.print(f"  Tenant:      [cyan]{cloud_tenant}[/cyan]")
        console.print(f"  Database:    [cyan]{cloud_database}[/cyan]")
    elif shared_path:
        console.print(f"  Mode:        [magenta]shared (filesystem)[/magenta]")
        console.print(f"  Shared:      [cyan]{shared_path}[/cyan]")
    if developer:
        console.print(f"  Developer:   [cyan]{developer}[/cyan]")
    console.print()
    console.print("Created files:")
    console.print(f"  .cognitive/manifest.json")
    console.print(f"  .cognitive/config.json")
    console.print(f"  .cognitive/llms.txt")
    console.print(f"  .cognitive/cached_profile.md")

    # If shared/cloud mode, immediately pull approved nodes
    if _shared_mode:
        console.print()
        console.print("[bold]Pulling existing team memories...[/bold]")
        try:
            from src.sync.sync import Syncer

            local_path = store_dir or str(Path.home() / ".cognitive-memory" / "store")
            cloud_api_key = os.environ.get("CMM_CHROMA_API_KEY")

            if _cloud_mode and not cloud_api_key:
                console.print("  [yellow]CMM_CHROMA_API_KEY not set — skipping pull.[/yellow]")
                console.print("  Set it and run [cyan]cmm pull[/cyan] to sync team memories.")
            else:
                cloud_creds = (
                    {"api_key": cloud_api_key, "tenant": cloud_tenant, "database": cloud_database}
                    if _cloud_mode else None
                )
                store = _make_store(local_path, shared_path, cloud_creds)
                syncer = Syncer(store=store, developer=developer or "")
                result = syncer.pull(proj.project_id, include_team=True)
                console.print(f"  [green]✓[/green] {result.summary}")

                profile = store.get_profile(proj.project_id)
                if profile:
                    from src.delivery.mcp_server import _fmt_profile
                    proj.update_cached_profile(_fmt_profile(profile))
                    console.print(f"  [green]✓[/green] Cached profile updated")
        except Exception as e:
            console.print(f"  [yellow]Pull skipped: {e}[/yellow]")

    console.print()
    if _cloud_mode:
        console.print("[dim]Ensure CMM_CHROMA_API_KEY is set before running cmm push/pull.[/dim]")
    else:
        console.print("[dim]Add .cognitive/ to .gitignore if you don't want it tracked.[/dim]")


@main.command()
@click.argument("node_id")
@click.option("--scope", type=click.Choice(["project", "team"]), required=True)
@click.option("--target", default=".", type=click.Path(exists=True))
@click.option("--project", "-p", default=None)
def classify(node_id, scope, target, project):
    """Reclassify a node's scope (project ↔ team)."""
    from rich.console import Console

    console = Console()
    project_id, local_path, shared_path, _, cloud_creds = _resolve_sync_paths(target, project)

    store = _make_store(local_path, shared_path, cloud_creds)
    # Look up the node in local first
    try:
        existing = store.nodes_col_local.get(ids=[node_id], include=["metadatas"])
        if not existing["ids"]:
            console.print(f"[red]Node {node_id} not found in local store.[/red]")
            return
        new_meta = dict(existing["metadatas"][0])
        old_scope = new_meta.get("scope", "project")
        new_meta["scope"] = scope
        store.nodes_col_local.update(ids=[node_id], metadatas=[new_meta])
        console.print(f"[green]✓[/green] Reclassified {node_id}: {old_scope} → {scope}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ── cmm sync ──────────────────────────────────────────────────────────────────

@main.command()
@click.argument("target", default=".", type=click.Path(exists=True))
def sync(target):
    """Update cached_profile.md and llms.txt from latest ChromaDB state.

    Reads the current cognitive profile from the store and writes it to
    .cognitive/cached_profile.md and regenerates .cognitive/llms.txt.
    """
    from pathlib import Path
    from rich.console import Console

    from src.discovery.project import CognitiveProject
    from src.discovery.llms_txt import generate_llms_txt
    from src.store.vector_store import MemoryStore
    from src.delivery.mcp_server import _fmt_profile

    console = Console()
    project_dir = Path(target).resolve()

    try:
        proj = CognitiveProject.load(project_dir)
    except FileNotFoundError:
        console.print("[red]No .cognitive/ folder found. Run 'cmm init' first.[/red]")
        return

    store_path = proj.config.get(
        "store_path",
        str(Path.home() / ".cognitive-memory" / "store"),
    )
    store = MemoryStore(persist_dir=store_path)
    profile = store.get_profile(proj.project_id)
    node_count = store.node_count(proj.project_id)

    if profile:
        # Update cached_profile.md
        profile_md = _fmt_profile(profile)
        proj.update_cached_profile(profile_md)

        # Regenerate llms.txt
        llms_content = generate_llms_txt(
            project_name=proj.name,
            project_description=proj.description,
            profile=profile,
            project_dir=project_dir,
        )
        proj.llms_txt_path.write_text(llms_content)

        console.print(f"[green]Synced:[/green] {proj.project_id}")
        console.print(f"  Nodes in store:       {node_count}")
        console.print(f"  Insights:             {len(profile.architectural_insights)}")
        console.print(f"  Pitfalls:             {len(profile.pitfalls)}")
        console.print(f"  Diagnostic strategies:{len(profile.diagnostic_strategies)}")
        console.print(f"  Updated: cached_profile.md, llms.txt")
    else:
        console.print(f"[yellow]No profile built for {proj.project_id}.[/yellow]")
        console.print(f"  Nodes in store: {node_count}")
        if node_count > 0:
            console.print("  Run 'cmm consolidate' to build a profile from stored nodes.")
        else:
            console.print("  Ingest some sessions first, then consolidate.")


# ── cmm status (updated) ─────────────────────────────────────────────────────

@main.command()
@click.argument("target", default=".", type=click.Path(exists=True))
@click.option("--project", "-p", default=None, help="Override project ID")
def status(target, project):
    """Show cognitive memory status for a project."""
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table

    from src.store.vector_store import MemoryStore

    console = Console()
    project_dir = Path(target).resolve()

    # Try to discover from .cognitive/ first
    project_id = project
    store_path = None

    try:
        from src.discovery.project import CognitiveProject
        proj = CognitiveProject.load(project_dir)
        project_id = project_id or proj.project_id
        store_path = proj.config.get("store_path")
        console.print(f"[green]Project:[/green] {proj.name} ({proj.project_id})")
        console.print(f"  Sessions:  {proj.session_count}")
        console.print(f"  Last:      {proj.last_session or 'never'}")
        console.print()
    except FileNotFoundError:
        if not project_id:
            console.print("[red]No .cognitive/ folder found and no --project given.[/red]")
            console.print("Run 'cmm init' or pass --project/-p.")
            return

    store_path = store_path or str(Path.home() / ".cognitive-memory" / "store")
    store = MemoryStore(persist_dir=store_path)

    table = Table(title=f"Memory Store: {project_id}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    count = store.node_count(project_id)
    table.add_row("Stored nodes (local)", str(count))

    profile = store.get_profile(project_id)
    if profile:
        table.add_row("Profile sessions", str(profile.session_count))
        table.add_row("Architectural insights", str(len(profile.architectural_insights)))
        table.add_row("Pitfalls", str(len(profile.pitfalls)))
        table.add_row("Diagnostic strategies", str(len(profile.diagnostic_strategies)))
        table.add_row("Key patterns", str(len(profile.key_patterns)))
        table.add_row("Last updated", profile.last_updated.strftime("%Y-%m-%d %H:%M"))
    else:
        table.add_row("Profile", "[yellow]not built yet[/yellow]")

    console.print(table)

    # ── Sync status (only if shared mode is configured) ────────────
    _, _, shared_path, developer, cloud_creds = _resolve_sync_paths(target, project_id)
    if shared_path or cloud_creds:
        try:
            from src.sync.sync import Syncer
            shared_store = _make_store(store_path, shared_path, cloud_creds)
            syncer = Syncer(store=shared_store, developer=developer)
            sync_status = syncer.status(project_id)

            sync_table = Table(title="Sync Status", show_header=True)
            sync_table.add_column("Metric", style="cyan")
            sync_table.add_column("Value")
            mode_label = "[magenta]cloud (Chroma Cloud)[/magenta]" if cloud_creds else "[green]shared (filesystem)[/green]"
            sync_table.add_row("Mode", mode_label)
            sync_table.add_row("Developer", developer or "[dim]unset[/dim]")
            sync_table.add_row("Unpushed local nodes", str(sync_status["unpushed_nodes"]))
            sync_table.add_row("Approved in shared", str(sync_status["shared_approved"]))
            sync_table.add_row("Pending review", str(sync_status["pending_review"]))
            sync_table.add_row("Last push", sync_status["last_push"] or "[dim]never[/dim]")
            sync_table.add_row("Last pull", sync_status["last_pull"] or "[dim]never[/dim]")
            console.print(sync_table)
        except Exception as e:
            console.print(f"[dim]Sync status unavailable: {e}[/dim]")


# ── cmm hook ──────────────────────────────────────────────────────────────────

@main.group()
def hook():
    """Run Claude Code hooks (start/stop)."""


@hook.command("start")
@click.argument("project_dir", default=".", type=click.Path(exists=True))
def hook_start(project_dir):
    """Session-start hook: load context from .cognitive/ folder."""
    from pathlib import Path
    from src.discovery.hooks import session_start_hook

    output = session_start_hook(Path(project_dir))
    print(output)


@hook.command("stop")
@click.argument("project_dir", default=".", type=click.Path(exists=True))
def hook_stop(project_dir):
    """Session-stop hook: ingest the just-completed session."""
    import json
    from pathlib import Path
    from src.discovery.hooks import session_stop_hook

    result = session_stop_hook(Path(project_dir))
    print(json.dumps(result, indent=2))


# ── Existing commands (unchanged) ─────────────────────────────────────────────

@main.command()
@click.argument("sessions", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--project", "-p", required=True, help="Project ID")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--build-profile", is_flag=True, help="Build cognitive profile after ingestion")
@click.option("--no-llm", is_flag=True, help="Use heuristic extraction (no API calls)")
def ingest(sessions, project, store_dir, build_profile, no_llm):
    """Ingest session files into the memory store."""
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.ingest import main as ingest_main

    sys.argv = ["ingest"] + list(sessions) + ["-p", project]
    if store_dir:
        sys.argv += ["--store-dir", store_dir]
    if build_profile:
        sys.argv.append("--build-profile")
    if no_llm:
        sys.argv.append("--no-llm")
    asyncio.run(ingest_main())


@main.command()
@click.option("--store-path", envvar="CMM_STORE_PATH",
              default=None, help="Memory store path (or set CMM_STORE_PATH)")
@click.option("--project", envvar="CMM_PROJECT_ID",
              default=None, help="Default project ID (or set CMM_PROJECT_ID)")
def serve(store_path, project):
    """Start the MCP server (stdio transport)."""
    import os
    if store_path:
        os.environ["CMM_STORE_PATH"] = store_path
    if project:
        os.environ["CMM_PROJECT_ID"] = project

    from src.delivery.mcp_server import run
    run()


@main.command()
@click.option("--project", "-p", required=True, help="Project ID")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--output", "-o", default="output/", help="Output directory")
@click.option("--format", "-f", "fmt", default="html",
              type=click.Choice(["html", "mermaid", "json", "all"]),
              help="Output format (default: html)")
def visualize(project, store_dir, output, fmt):
    """Generate an interactive DAG visualization."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.visualize_dag import main as viz_main
    sys.argv = ["visualize", "-p", project, "-o", output, "-f", fmt]
    if store_dir:
        sys.argv += ["--store", store_dir]
    viz_main()


@main.command()
@click.option("--projects-dir", default=None, help="Claude Code projects directory")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--poll-interval", type=float, default=10.0, help="Seconds between polls")
@click.option("--min-age", type=float, default=30.0, help="Min file age before processing")
@click.option("--no-auto-ingest", is_flag=True, help="Only detect, don't auto-ingest")
def watch(projects_dir, store_dir, poll_interval, min_age, no_auto_ingest):
    """Watch for new Claude Code sessions and auto-ingest them."""
    import asyncio
    from pathlib import Path
    from src.ingestion.watcher import SessionWatcher

    store_path = store_dir or str(Path(__file__).parent.parent / "data" / "memory_store")
    watcher = SessionWatcher(
        watch_dir=projects_dir,
        store_path=store_path,
        poll_interval=poll_interval,
        min_file_age=min_age,
        auto_ingest=not no_auto_ingest,
    )
    asyncio.run(watcher.watch())


@main.command()
@click.option("--project", "-p", default=None, help="Project ID (or --all)")
@click.option("--all", "all_projects", is_flag=True, help="Consolidate all projects")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--upgrade", is_flag=True, help="Re-extract warm-tier nodes with LLM")
@click.option("--profiles-only", is_flag=True, help="Only rebuild profiles")
@click.option("--dry-run", is_flag=True, help="Show what would be processed")
def consolidate(project, all_projects, store_dir, upgrade, profiles_only, dry_run):
    """Run batch consolidation — rebuild profiles and optionally upgrade warm nodes."""
    import asyncio
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.batch_consolidate import main as consolidate_main

    argv = ["consolidate"]
    if project:
        argv += ["-p", project]
    if all_projects:
        argv.append("--all")
    if store_dir:
        argv += ["--store-dir", store_dir]
    if upgrade:
        argv.append("--upgrade")
    if profiles_only:
        argv.append("--profiles-only")
    if dry_run:
        argv.append("--dry-run")

    sys.argv = argv
    asyncio.run(consolidate_main())


@main.command()
@click.argument("target", type=click.Path(exists=True))
@click.option("--project", "-p", required=True, help="Project ID")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--python", default=None, help="Path to Python with cmm deps")
def install(target, project, store_dir, python):
    """Install cognitive memory skills into a project's .claude/commands/."""
    from pathlib import Path
    from scripts.install_skills import install as do_install, CMM_ROOT, DEFAULT_STORE

    python_path = Path(python) if python else CMM_ROOT / ".venv" / "bin" / "python"
    store_path = Path(store_dir) if store_dir else DEFAULT_STORE
    do_install(Path(target), project, store_path, python_path)


# ══════════════════════════════════════════════════════════════════════
#  Phase 4: Push / Pull / Status (shared store sync)
# ══════════════════════════════════════════════════════════════════════


def _resolve_sync_paths(target, project_id_arg):
    """Resolve (project_id, local_path, shared_path, developer, cloud_creds) from
    .cognitive/config.json + env var overrides.

    cloud_creds is a dict with keys api_key, tenant, database when Chroma Cloud
    is configured (tenant+database present + CMM_CHROMA_API_KEY set), else None.
    When cloud_creds is returned, callers should pass it to MemoryStore and may
    pass shared_path=None.
    """
    from pathlib import Path
    import os, json

    target = Path(target).resolve()
    cfg_path = target / ".cognitive" / "config.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}

    project_id = project_id_arg or os.environ.get("CMM_PROJECT_ID") or cfg.get("project_id")
    local_path = (
        os.environ.get("CMM_STORE_PATH")
        or cfg.get("local_store_path")
        or str(Path.home() / ".cognitive-memory" / "store")
    )
    shared_path = (
        os.environ.get("CMM_SHARED_STORE_PATH")
        or cfg.get("shared_store_path")
    )
    developer = (
        os.environ.get("CMM_DEVELOPER_NAME")
        or cfg.get("developer_name")
        or ""
    )

    # Chroma Cloud credentials — API key from env only (never stored in config)
    cloud_tenant = os.environ.get("CMM_CHROMA_TENANT") or cfg.get("cloud_tenant")
    cloud_database = os.environ.get("CMM_CHROMA_DATABASE") or cfg.get("cloud_database")
    cloud_api_key = os.environ.get("CMM_CHROMA_API_KEY")  # env only

    cloud_creds = None
    if cloud_tenant and cloud_database and cloud_api_key:
        cloud_creds = {
            "api_key": cloud_api_key,
            "tenant": cloud_tenant,
            "database": cloud_database,
        }

    return project_id, local_path, shared_path, developer, cloud_creds


def _make_store(local_path, shared_path, cloud_creds):
    """Construct a MemoryStore, using Chroma Cloud if cloud_creds provided."""
    from src.store.vector_store import MemoryStore

    if cloud_creds:
        return MemoryStore(
            local_path=local_path,
            cloud_api_key=cloud_creds["api_key"],
            cloud_tenant=cloud_creds["tenant"],
            cloud_database=cloud_creds["database"],
        )
    return MemoryStore(local_path=local_path, shared_path=shared_path)


@main.command()
@click.option("--project", "-p", default=None, help="Project ID (overrides config)")
@click.option("--target", default=".", type=click.Path(exists=True), help="Project directory")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed without pushing")
def push(project, target, dry_run):
    """Push new local nodes to the shared staging area for review."""
    from rich.console import Console
    from src.sync.sync import Syncer

    console = Console()
    project_id, local_path, shared_path, developer, cloud_creds = _resolve_sync_paths(target, project)

    if not project_id:
        console.print("[red]No project ID. Pass --project or set CMM_PROJECT_ID.[/red]")
        return
    if not shared_path and not cloud_creds:
        console.print("[red]No shared store configured. Set CMM_CHROMA_TENANT/CMM_CHROMA_DATABASE/CMM_CHROMA_API_KEY for cloud, or CMM_SHARED_STORE_PATH for filesystem.[/red]")
        return

    store = _make_store(local_path, shared_path, cloud_creds)
    syncer = Syncer(store=store, developer=developer)
    result = syncer.push(project_id, dry_run=dry_run)

    if result.errors:
        for e in result.errors:
            console.print(f"[red]Error: {e}[/red]")
        return

    if dry_run:
        console.print(f"[yellow]DRY RUN[/yellow] would push [cyan]{result.pushed}[/cyan] nodes")
    else:
        console.print(f"[green]✓[/green] {result.summary}")
        console.print(f"  Reviewers can run [cyan]cmm review --project {project_id}[/cyan]")


@main.command()
@click.option("--project", "-p", default=None, help="Project ID (overrides config)")
@click.option("--target", default=".", type=click.Path(exists=True), help="Project directory")
@click.option("--pending-count", is_flag=True, help="Just print the pending count and exit")
def review(project, target, pending_count):
    """Review and approve/reject staged shared-store nodes."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from src.sync.review import Reviewer, ReviewAction, ReviewDecision

    console = Console()
    project_id, local_path, shared_path, developer, cloud_creds = _resolve_sync_paths(target, project)

    if not shared_path and not cloud_creds:
        console.print("[red]No shared store configured. Set CMM_CHROMA_TENANT/CMM_CHROMA_DATABASE/CMM_CHROMA_API_KEY for cloud, or CMM_SHARED_STORE_PATH for filesystem.[/red]")
        return

    store = _make_store(local_path, shared_path, cloud_creds)
    reviewer = Reviewer(store=store, reviewer_name=developer or "anonymous")

    if pending_count:
        n = reviewer.pending_count(project_id)
        scope = f" for {project_id}" if project_id else ""
        console.print(f"{n} nodes pending review{scope}")
        return

    pending = reviewer.list_pending(project_id)
    if not pending:
        console.print("[green]No pending nodes to review.[/green]")
        return

    console.print(Panel(
        f"[bold]Cognitive Memory Review[/bold]\n"
        f"Project: [cyan]{project_id or 'all projects'}[/cyan]\n"
        f"Pending: [yellow]{len(pending)}[/yellow] nodes\n"
        f"Reviewer: [cyan]{developer or 'anonymous'}[/cyan]",
        style="blue",
    ))

    def _prompt_for(node, idx, total):
        while True:
            console.rule(f"[bold cyan]Node {idx + 1} / {total}[/bold cyan]")
            ntype = (node.get("node_type") or "?").upper()
            scope = node.get("scope", "project")
            conf = node.get("confidence", 0)
            session = node.get("session_id", "?")
            source = node.get("source_developer") or "unknown"
            console.print(
                f"[bold]{ntype}[/bold]   "
                f"scope=[{'magenta' if scope == 'team' else 'green'}]{scope}[/]   "
                f"confidence=[yellow]{conf:.2f}[/]   "
                f"by [cyan]{source}[/cyan]   "
                f"session=[dim]{session}[/dim]"
            )
            console.print()
            console.print(f"[bold]Summary:[/bold] {node.get('summary', '')}")
            evidence = node.get("evidence", "")
            if evidence:
                console.print(f"[bold]Evidence:[/bold] [dim]{evidence[:300]}[/dim]")
            console.print()

            choice = Prompt.ask(
                "[a]pprove  [r]eject  [s]wap-scope  [e]dit  [k]skip  [q]uit",
                choices=["a", "r", "s", "e", "k", "q"],
                default="a",
            )

            if choice == "a":
                return ReviewDecision(action=ReviewAction.APPROVE)
            if choice == "r":
                reason = Prompt.ask("Rejection reason (optional)", default="")
                return ReviewDecision(action=ReviewAction.REJECT, reason=reason)
            if choice == "s":
                new_scope = "team" if scope == "project" else "project"
                console.print(f"[dim]Scope flipped: {scope} → {new_scope}[/dim]")
                return ReviewDecision(action=ReviewAction.SWAP_SCOPE, new_scope=new_scope)
            if choice == "e":
                new_summary = Prompt.ask("New summary", default=node.get("summary", ""))
                return ReviewDecision(action=ReviewAction.EDIT_SUMMARY, new_summary=new_summary)
            if choice == "k":
                return ReviewDecision(action=ReviewAction.SKIP)
            if choice == "q":
                return ReviewDecision(action=ReviewAction.QUIT)

    summary = reviewer.review(project_id, _prompt_for)

    console.print()
    console.rule()
    console.print(f"[bold green]✓ {summary.text}[/bold green]")
    if summary.scope_changes:
        console.print(f"  ({summary.scope_changes} scope changes, {summary.summary_edits} summary edits)")


@main.command()
@click.option("--project", "-p", default=None, help="Project ID (overrides config)")
@click.option("--target", default=".", type=click.Path(exists=True), help="Project directory")
@click.option("--no-team", is_flag=True, help="Skip team-scope nodes")
def pull(project, target, no_team):
    """Pull approved nodes from the shared store into the local cache."""
    from rich.console import Console
    from src.sync.sync import Syncer

    console = Console()
    project_id, local_path, shared_path, developer, cloud_creds = _resolve_sync_paths(target, project)

    if not project_id:
        console.print("[red]No project ID. Pass --project or set CMM_PROJECT_ID.[/red]")
        return
    if not shared_path and not cloud_creds:
        console.print("[red]No shared store configured. Set CMM_CHROMA_TENANT/CMM_CHROMA_DATABASE/CMM_CHROMA_API_KEY for cloud, or CMM_SHARED_STORE_PATH for filesystem.[/red]")
        return

    store = _make_store(local_path, shared_path, cloud_creds)
    syncer = Syncer(store=store, developer=developer)
    result = syncer.pull(project_id, include_team=not no_team)

    if result.errors:
        for e in result.errors:
            console.print(f"[red]Error: {e}[/red]")
        return

    console.print(f"[green]✓[/green] {result.summary}")
