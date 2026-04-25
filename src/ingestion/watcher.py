"""Session watcher daemon — watches for new/modified Claude Code sessions and triggers ingestion.

The watcher polls ~/.claude/projects/ for JSONL session files. When it detects
a new or modified file, it runs warm-tier heuristic extraction immediately and
optionally queues full LLM extraction for later batch processing.

Usage:
    # As a module
    python -m src.ingestion.watcher --projects-dir ~/.claude/projects/

    # Programmatically
    watcher = SessionWatcher(store=store, project_map={"supply-chain": "/path/to/project"})
    await watcher.watch()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WatchedFile:
    """Track state of a watched session file."""
    path: Path
    last_modified: float
    last_size: int
    ingested: bool = False
    ingest_time: float = 0.0


@dataclass
class WatchEvent:
    """Emitted when a session file changes."""
    path: Path
    project_dir_name: str
    event_type: str  # "new" or "modified"
    file_size: int


class SessionWatcher:
    """Watch for completed Claude Code sessions and trigger ingestion.

    The watcher maintains a state file (.cmm_watcher_state.json) in the store
    directory to persist which files have been processed across restarts.

    Parameters:
        watch_dir: Directory to watch (default: ~/.claude/projects/)
        store_path: Path to the ChromaDB memory store
        poll_interval: Seconds between polls (default: 10)
        min_file_age: Seconds a file must be unmodified before processing (default: 30)
            Prevents ingesting sessions that are still being written to.
        on_new_session: Async callback invoked when a new/modified session is detected.
            Signature: async (event: WatchEvent) -> None
        auto_ingest: If True, automatically run warm-tier extraction on detected sessions.
        project_map: Optional mapping of project_dir_name → project_id.
            If not provided, the watcher derives project_id from the directory name.
    """

    def __init__(
        self,
        watch_dir: str | Path | None = None,
        store_path: str | Path | None = None,
        poll_interval: float = 10.0,
        min_file_age: float = 30.0,
        on_new_session=None,
        auto_ingest: bool = True,
        project_map: dict[str, str] | None = None,
    ):
        self.watch_dir = Path(watch_dir or os.path.expanduser("~/.claude/projects"))
        self.store_path = store_path
        self.poll_interval = poll_interval
        self.min_file_age = min_file_age
        self.on_new_session = on_new_session
        self.auto_ingest = auto_ingest
        self.project_map = project_map or {}

        self._tracked: dict[str, WatchedFile] = {}
        self._running = False
        self._state_file = Path(store_path) / ".cmm_watcher_state.json" if store_path else None

    # ── State persistence ─────────────────────────────────────────

    def _load_state(self):
        """Load previously processed files from state file."""
        if not self._state_file or not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
            for entry in data.get("processed", []):
                path = Path(entry["path"])
                self._tracked[str(path)] = WatchedFile(
                    path=path,
                    last_modified=entry["last_modified"],
                    last_size=entry["last_size"],
                    ingested=True,
                    ingest_time=entry.get("ingest_time", 0),
                )
            logger.info("Loaded state: %d previously processed files", len(self._tracked))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load watcher state: %s", e)

    def _save_state(self):
        """Persist processed files to state file."""
        if not self._state_file:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        processed = [
            {
                "path": str(wf.path),
                "last_modified": wf.last_modified,
                "last_size": wf.last_size,
                "ingest_time": wf.ingest_time,
            }
            for wf in self._tracked.values()
            if wf.ingested
        ]
        self._state_file.write_text(json.dumps({"processed": processed}, indent=2))

    # ── Directory scanning ────────────────────────────────────────

    def scan(self) -> list[WatchEvent]:
        """Scan for new or modified session files. Returns list of events."""
        events = []
        now = time.time()

        if not self.watch_dir.exists():
            logger.warning("Watch directory does not exist: %s", self.watch_dir)
            return events

        # Iterate project directories
        for project_dir in self.watch_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Find JSONL session files
            for jsonl_file in project_dir.glob("*.jsonl"):
                key = str(jsonl_file)
                stat = jsonl_file.stat()
                mtime = stat.st_mtime
                size = stat.st_size

                # Skip empty files
                if size == 0:
                    continue

                # Skip files still being written to (modified too recently)
                age = now - mtime
                if 0 < age < self.min_file_age:
                    continue

                tracked = self._tracked.get(key)

                if tracked is None:
                    # New file
                    self._tracked[key] = WatchedFile(
                        path=jsonl_file,
                        last_modified=mtime,
                        last_size=size,
                    )
                    events.append(WatchEvent(
                        path=jsonl_file,
                        project_dir_name=project_dir.name,
                        event_type="new",
                        file_size=size,
                    ))
                elif not tracked.ingested and (mtime > tracked.last_modified or size != tracked.last_size):
                    # Modified since last scan, not yet ingested
                    tracked.last_modified = mtime
                    tracked.last_size = size
                    events.append(WatchEvent(
                        path=jsonl_file,
                        project_dir_name=project_dir.name,
                        event_type="modified",
                        file_size=size,
                    ))
                elif tracked.ingested and mtime > tracked.ingest_time:
                    # File was modified after we ingested it (session continued)
                    tracked.last_modified = mtime
                    tracked.last_size = size
                    tracked.ingested = False
                    events.append(WatchEvent(
                        path=jsonl_file,
                        project_dir_name=project_dir.name,
                        event_type="modified",
                        file_size=size,
                    ))

        return events

    def mark_ingested(self, path: Path):
        """Mark a file as successfully ingested."""
        key = str(path)
        if key in self._tracked:
            self._tracked[key].ingested = True
            self._tracked[key].ingest_time = time.time()
            self._save_state()

    # ── Project ID derivation ─────────────────────────────────────

    def derive_project_id(self, project_dir_name: str) -> str:
        """Derive a project ID from the Claude Code directory name.

        Claude Code encodes the working directory by replacing both '/' and
        '.' with '-', so dir names are ambiguous (e.g. the encoded name
        `-home-joelaf-repo-linux-nova-irq-7-1` could come from
        `/home/joelaf/repo/linux-nova-irq-7.1` or
        `/home/joelaf/repo/linux/nova/irq/7/1`). The only reliable source
        for the original cwd is the ``cwd`` field inside the session JSONL
        files, so we prefer reading that when available.

        Strategy:
            1. If an explicit project_map entry exists, honour it.
            2. Scan any session JSONL in the dir for a line containing
               ``cwd`` and return basename(cwd).
            3. Fall back to the legacy skip-common-segments heuristic if
               no JSONL contains a cwd (empty dir, snapshot-only files).
        """
        if project_dir_name in self.project_map:
            return self.project_map[project_dir_name]

        # Authoritative: pull the original cwd from any session file
        project_dir = self.watch_dir / project_dir_name
        try:
            for jsonl in project_dir.glob("*.jsonl"):
                try:
                    with jsonl.open() as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            cwd = entry.get("cwd")
                            if cwd:
                                name = os.path.basename(cwd.rstrip("/"))
                                if name:
                                    return name
                except OSError:
                    continue
        except OSError:
            pass

        # Fallback: original heuristic for dirs without readable session files
        parts = project_dir_name.strip("-").split("-")
        skip = {"users", "home", "documents", "downloads", "desktop", "projects", "repos", "src", "code"}
        meaningful = [p for p in parts if p.lower() not in skip]
        return meaningful[-1] if meaningful else project_dir_name

    # ── Auto-ingest ───────────────────────────────────────────────

    async def _auto_ingest(self, event: WatchEvent):
        """Run warm-tier extraction on a detected session."""
        from src.ingestion import ClaudeCodeParser
        from src.extraction.warm_extractor import WarmExtractor
        from src.store import MemoryStore
        from src.compression import SemanticDeduplicator

        project_id = self.derive_project_id(event.project_dir_name)
        logger.info("Auto-ingesting %s → project:%s", event.path.name, project_id)

        try:
            parser = ClaudeCodeParser()
            session = parser.parse_file(event.path)

            extractor = WarmExtractor()
            dag = extractor.extract(session)

            store = MemoryStore(persist_dir=str(self.store_path))
            dedup = SemanticDeduplicator(store)
            result = dedup.deduplicate(dag.nodes, project_id, session.session_id)

            from src.schemas.reasoning import ReasoningDAG
            dag_to_store = ReasoningDAG(
                session_id=dag.session_id,
                nodes=result.stored,
                edges=dag.edges,
                pivot_nodes=dag.pivot_nodes,
                noise_ratio=dag.noise_ratio,
            )
            stored = store.store_dag(dag_to_store, project_id)

            self.mark_ingested(event.path)
            logger.info(
                "Ingested %s: %d messages → %d nodes → %d stored (%d merged, %d dropped)",
                event.path.name, len(session.messages), len(dag.nodes),
                stored, len(result.merged), len(result.dropped),
            )
        except Exception as e:
            logger.error("Failed to ingest %s: %s", event.path.name, e)

    # ── Main watch loop ───────────────────────────────────────────

    async def watch(self):
        """Start the watch loop. Runs until stop() is called or KeyboardInterrupt."""
        self._running = True
        self._load_state()
        logger.info("Watching %s (poll every %.0fs, min age %.0fs)", self.watch_dir, self.poll_interval, self.min_file_age)

        try:
            while self._running:
                events = self.scan()

                for event in events:
                    logger.info("[%s] %s (%d bytes)", event.event_type.upper(), event.path.name, event.file_size)

                    if self.on_new_session:
                        await self.on_new_session(event)

                    if self.auto_ingest and self.store_path:
                        await self._auto_ingest(event)

                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("Watcher cancelled")
        finally:
            self._running = False
            self._save_state()

    def stop(self):
        """Signal the watch loop to stop."""
        self._running = False


# ── CLI entry point ───────────────────────────────────────────────

async def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Watch for new Claude Code sessions")
    parser.add_argument(
        "--projects-dir",
        default=os.path.expanduser("~/.claude/projects"),
        help="Claude Code projects directory (default: ~/.claude/projects/)",
    )
    parser.add_argument(
        "--store",
        default=str(Path(__file__).parent.parent.parent / "data" / "memory_store"),
        help="Memory store directory",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=10.0,
        help="Seconds between polls (default: 10)",
    )
    parser.add_argument(
        "--min-age", type=float, default=30.0,
        help="Seconds a file must be unmodified before processing (default: 30)",
    )
    parser.add_argument(
        "--no-auto-ingest", action="store_true",
        help="Only detect sessions, don't auto-ingest",
    )
    parser.add_argument(
        "--map", nargs=2, action="append", metavar=("DIR_NAME", "PROJECT_ID"),
        help="Map a directory name to a project ID (can repeat)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    project_map = {}
    if args.map:
        for dir_name, project_id in args.map:
            project_map[dir_name] = project_id

    watcher = SessionWatcher(
        watch_dir=args.projects_dir,
        store_path=args.store,
        poll_interval=args.poll_interval,
        min_file_age=args.min_age,
        auto_ingest=not args.no_auto_ingest,
        project_map=project_map,
    )

    await watcher.watch()


if __name__ == "__main__":
    asyncio.run(_main())
