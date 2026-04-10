# Cognitive Memory Manager (CMM)

Persistent reasoning memory for AI coding agents. CMM captures reasoning
patterns from Claude Code sessions, consolidates them into durable knowledge
(architectural insights, known pitfalls, diagnostic strategies), and
retrieves them in future sessions — so agents stop rediscovering the same
things.

For teams, a distributed store with human-gated approval lets developers
share knowledge across projects without one agent's mistake becoming
institutional false knowledge.

## Quickstart (2 minutes)

```bash
# Clone and install
git clone https://github.com/sazandkhalid/cmm.git
cd cmm/cognitive-memory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Set your OpenAI API key (needed for embeddings)
export OPENAI_API_KEY="sk-..."

# Run the demo — ingests sample sessions, builds a profile, queries it
python scripts/demo.py
```

The demo ingests 10 included fixture sessions, extracts reasoning DAGs,
builds a cognitive profile, runs a search query, and generates an
interactive HTML visualization — all in one command.

## Prerequisites

- **Python 3.12+**
- **OpenAI API key** — used for `text-embedding-3-small` embeddings
  (required for ingestion, search, and profile building)
- **Anthropic API key** (optional) — used for cold-tier LLM extraction
  and profile building. Without it, the system uses warm-tier heuristic
  extraction only (still useful, just lower confidence).

## Installation

```bash
cd cognitive-memory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # includes pytest
```

This registers the `cmm` CLI command. Verify with:

```bash
cmm --help
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional — enables cold-tier LLM extraction + profile building
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional — defaults shown
export CMM_STORE_PATH="~/.cognitive-memory/store"
export CMM_PROJECT_ID="my-project"
export CMM_CONTEXT_FILL_RATIO="0.45"

# For shared/team mode
export CMM_SHARED_STORE_PATH="/shared/team/memory_store"
export CMM_DEVELOPER_NAME="yourname"
```

## How It Works

### Pipeline

```
Session JSONL → Parse → Extract reasoning DAG → Deduplicate → Store in ChromaDB
    → Cluster → Build cognitive profile → Serve via MCP / CLI / hooks
```

### Two Extraction Tiers

| Tier | Speed | API Cost | Confidence | When |
|------|-------|----------|------------|------|
| **Warm** | < 1 second | Free | 0.25–0.75 | Every session end (automatic) |
| **Cold** | ~30 seconds | Anthropic API | 0.50–0.95 | On demand (`cmm consolidate --upgrade`) |

The warm tier uses regex heuristics (keyword scoring, error-resolution
pairing, explicit conclusion detection). The cold tier packs session
messages into token-budget windows at 45% of the model's context capacity
and asks Claude to classify each reasoning step.

### Reasoning DAG

Each session becomes a directed graph of typed reasoning nodes:

- **HYPOTHESIS** — agent forms a theory
- **INVESTIGATION** — agent examines evidence
- **DISCOVERY** — something unexpected found
- **PIVOT** — agent changes approach
- **DEAD_END** — approach that failed
- **SOLUTION** — working resolution

Edges capture causal relationships: `led_to`, `refined`, `caused_pivot_to`.

### Cognitive Profile

Clustering + LLM classification consolidates raw nodes into:

- **Architectural insights** — structural knowledge about the codebase
- **Known pitfalls** — ranked by severity, with resolution strategies
- **Diagnostic strategies** — proven debugging approaches with success rates
- **Key patterns** and **anti-patterns**

## Usage

### Single Developer (local mode)

```bash
# Initialize cognitive memory for a project
cd /path/to/your/project
cmm init .

# Ingest a session transcript
cmm ingest ~/.claude/projects/-path-to-project/session-id.jsonl \
    --project my-project --build-profile

# Or ingest without LLM (fast, free)
cmm ingest session.jsonl --project my-project --no-llm

# Query the memory
cmm serve                    # Start MCP server for Claude Code
cmm status                   # Show memory stats
```

### Using as Claude Code Skills

After init, Claude Code can use these slash commands:

| Command | What it does |
|---------|-------------|
| `/cognitive-profile` | Load the full distilled profile |
| `/search-memory` | Semantic search over past reasoning |
| `/pitfalls` | Ranked list of known traps |
| `/diagnose` | Proven debugging strategies for a problem |
| `/consolidate` | Trigger cold-tier profile rebuild |
| `/visualize-dag` | Interactive reasoning graph |

### Team Mode (shared store)

```bash
# Developer 1: Initialize with shared store
cmm init . --shared /shared/team/memory --developer alice

# Push new local memories to shared staging
cmm push --project my-project

# Reviewer: approve or reject staged memories
cmm review --project my-project

# Developer 2: Pull approved memories (including team-scope)
cmm pull --project my-project

# Check sync status
cmm status
```

The shared store has a staging area with human-in-the-loop review:
pushed nodes must be approved before they become team-visible.

### Memory Scope

Every memory is classified as:

- **PROJECT** — specific to one repo (e.g., "Alembic needs models in `__init__.py`")
- **TEAM** — general knowledge (e.g., "Pydantic v2 deprecated `class Config`")

Team-scope memories cross project boundaries automatically on `cmm pull`.

## CLI Reference

```
cmm init [dir]                Initialize .cognitive/ folder
    --shared PATH             Enable shared mode + auto-pull
    --developer NAME          Your name for attribution
    --team-id ID              Team identifier

cmm status [dir]              Show memory stats + sync status
cmm sync [dir]                Update cached_profile.md from store

cmm ingest <files...>         Ingest JSONL session files
    --project ID
    --build-profile           Build profile after ingestion
    --no-llm                  Heuristic extraction only (free)

cmm consolidate               Batch rebuild profiles
    --project ID | --all
    --upgrade                 Re-extract warm nodes with LLM

cmm push                      Push to shared staging
cmm pull                      Pull approved from shared
cmm review                    Interactive review UI
    --pending-count           Just print count

cmm classify NODE_ID          Reclassify scope
    --scope project|team

cmm serve                     Start MCP server
cmm watch                     Start session watcher daemon
cmm visualize --project ID    Generate interactive DAG HTML
cmm install TARGET -p ID      Install skills to .claude/commands/
```

## Running Tests

```bash
# All tests (no API keys needed — embeddings are mocked)
pytest tests/ -q

# With coverage
pytest tests/ --cov=src/ --cov-report=term-missing

# A specific test file
pytest tests/test_token_windowing.py -v
```

The test suite (193 tests) mocks all external API calls. No OpenAI or
Anthropic keys are needed to run tests.

## Project Structure

```
cognitive-memory/
├── src/
│   ├── schemas/              # Pydantic models (session, reasoning, memory)
│   ├── ingestion/            # JSONL parsing + session watcher
│   ├── extraction/           # Warm (heuristic) + cold (LLM) extraction
│   ├── compression/          # Semantic dedup + profile building
│   ├── store/                # ChromaDB vector store (local + shared)
│   ├── delivery/             # MCP server + CLI query interface
│   ├── discovery/            # .cognitive/ folder + hooks + llms.txt
│   ├── evaluation/           # Session analysis + interaction logging
│   ├── sync/                 # Push/pull/review for shared mode
│   └── cli.py                # Click CLI entry point
├── scripts/
│   ├── demo.py               # End-to-end demo on fixture data
│   ├── ingest.py             # Batch ingestion
│   ├── batch_consolidate.py  # Cold-tier consolidation
│   ├── controlled_comparison.py  # A/B memory evaluation
│   ├── eval_report.py        # Evaluation dashboard
│   └── visualize_dag.py      # DAG visualization generator
├── fixtures/                 # 10 sample session transcripts
├── tests/                    # 193 tests
├── ARCHITECTURE.md           # Comprehensive architecture documentation
└── pyproject.toml
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system documentation
including data flow diagrams, module dependency graph, schema details,
and configuration reference.

## License

MIT
