# Cognitive Memory Manager (CMM)

Persistent reasoning memory for AI coding agents. CMM captures reasoning
patterns from coding sessions, consolidates them into durable knowledge
(architectural insights, known pitfalls, diagnostic strategies), and
retrieves them in future sessions -- so agents stop rediscovering the same
things.

## Why CMM?

AI coding assistants are powerful but forgetful. Every new session starts
from scratch -- the agent re-discovers the same codebase quirks, falls into
the same traps, and repeats the same debugging steps that another team
member's agent already solved last week. This is wasted time and wasted
tokens.

CMM solves this by building a **shared team memory** that learns from every
coding session and makes that knowledge available to every future session.

**The problem CMM addresses:**

- **Agents repeat mistakes.** Without persistent memory, an AI assistant
  that spent 20 minutes debugging a Pydantic v2 migration issue will hit
  the exact same wall next session. Multiply that across a team of 10
  developers and the waste compounds fast.

- **Knowledge stays siloed.** When one developer's agent discovers that
  "Alembic needs models registered in `__init__.py` to detect migrations,"
  that insight dies with the session. Other team members hit the same
  issue independently.

- **Onboarding is slow.** New team members start with zero context. Their
  AI assistant has no awareness of project conventions, known pitfalls, or
  proven debugging strategies that the rest of the team has already
  internalized.

**What CMM does about it:**

- **Captures reasoning patterns** from every coding session -- not just
  what happened, but how the agent reasoned about it: hypotheses formed,
  investigations performed, dead ends hit, pivots taken, solutions found.

- **Builds a shared knowledge base** that improves over time. The more
  sessions your team runs, the smarter every future session becomes. Known
  pitfalls get surfaced before you hit them. Proven debugging strategies
  get recommended when similar problems appear.

- **Human-gated quality control.** Pushed memories go through a staging
  area with human review before becoming team-visible. One agent's
  hallucination doesn't become institutional false knowledge.

- **AI assistant agnostic.** CMM works with any AI coding assistant that
  produces session transcripts -- Claude Code, Cursor, Windsurf, or any
  future tool. The memory layer is independent of the model powering your
  assistant. Switch models or tools without losing your accumulated
  knowledge.

- **Cross-project learning.** Knowledge is classified as project-specific
  or team-general. General insights (like "Pydantic v2 deprecated
  `class Config`") automatically propagate across all your projects.

**The result:** Your team's AI assistants get progressively better at your
codebase. Onboarding drops from days to hours because new developers
inherit the entire team's debugging history. Recurring pitfalls get caught
before they waste anyone's time.

## Quickstart (2 minutes)

```bash
# Clone and install
git clone https://github.com/sazandkhalid/cmm.git
cd cmm
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set your OpenAI API key (needed for embeddings)
export OPENAI_API_KEY="sk-..."

# Run the demo -- ingests sample sessions, builds a profile, queries it
uv run python cognitive-memory/scripts/demo.py
```

The demo ingests 10 included fixture sessions, extracts reasoning DAGs,
builds a cognitive profile, runs a search query, and generates an
interactive HTML visualization -- all in one command.

No API keys? Run parse-only mode (zero API calls):

```bash
uv run python cognitive-memory/scripts/demo.py --parse-only
```

Clean up demo artifacts afterward:

```bash
uv run python cognitive-memory/scripts/demo.py --clean
```

## Prerequisites

- **Python 3.12+**
- **OpenAI API key** -- used for `text-embedding-3-small` embeddings
  (required for ingestion, search, and profile building)
- **LLM provider** (optional) -- used for cold-tier LLM extraction
  (Claude Sonnet 4.5/4.6) and profile building. Supports either direct
  Anthropic API or Amazon Bedrock. Without it, the system uses warm-tier
  heuristic extraction only (still useful, just lower confidence).

## Installation

```bash
cd cmm
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"     # includes pytest
```

This registers the `cmm` CLI command. Verify with:

```bash
cmm --help
```

### Environment Variables

See [`.env.example`](cognitive-memory/.env.example) for a complete template.

```bash
# Required
export OPENAI_API_KEY="sk-..."

# LLM provider -- choose one:
# Option A: Direct Anthropic API
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B: Amazon Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# Optional: override the LLM model (default: auto-detect from credentials)
# export CMM_LLM_MODEL="anthropic/claude-sonnet-4-6"                        # direct API
# export CMM_LLM_MODEL="bedrock/anthropic.claude-sonnet-4-6"                # Bedrock
# export CMM_LLM_MODEL="bedrock/anthropic.claude-opus-4-6-v1"               # Bedrock Opus

# Optional -- defaults shown
export CMM_STORE_PATH="~/.cognitive-memory/store"
export CMM_PROJECT_ID="my-project"
export CMM_CONTEXT_FILL_RATIO="0.45"

# For shared/team mode -- Chroma Cloud (recommended)
export CMM_CHROMA_API_KEY="ck-..."          # Chroma Cloud API key
export CMM_CHROMA_TENANT="<tenant-uuid>"    # Chroma Cloud tenant ID
export CMM_CHROMA_DATABASE="cmm"            # Chroma Cloud database name
export CMM_DEVELOPER_NAME="yourname"

# For shared/team mode -- filesystem (self-hosted alternative)
export CMM_SHARED_STORE_PATH="/shared/team/memory_store"
export CMM_DEVELOPER_NAME="yourname"
```

## How It Works

### Pipeline

```
Session JSONL -> Parse -> Extract reasoning DAG -> Deduplicate -> Store in ChromaDB
    -> Cluster -> Build cognitive profile -> Serve via MCP / CLI / hooks
```

### Two Extraction Tiers

| Tier | Speed | API Cost | Confidence | When |
|------|-------|----------|------------|------|
| **Warm** | < 1 second | Free | 0.25-0.75 | Every session end (automatic) |
| **Cold** | ~30 seconds | LLM API | 0.50-0.95 | On demand (`cmm consolidate --upgrade`) |

The warm tier uses regex heuristics (keyword scoring, error-resolution
pairing, explicit conclusion detection). The cold tier packs session
messages into token-budget windows at 45% of the model's context capacity
and asks Claude to classify each reasoning step.

### Reasoning DAG

Each session becomes a directed graph of typed reasoning nodes:

- **HYPOTHESIS** -- agent forms a theory
- **INVESTIGATION** -- agent examines evidence
- **DISCOVERY** -- something unexpected found
- **PIVOT** -- agent changes approach
- **DEAD_END** -- approach that failed
- **SOLUTION** -- working resolution

Edges capture causal relationships: `led_to`, `refined`, `caused_pivot_to`.

### Cognitive Profile

Clustering + LLM classification consolidates raw nodes into:

- **Architectural insights** -- structural knowledge about the codebase
- **Known pitfalls** -- ranked by severity, with resolution strategies
- **Diagnostic strategies** -- proven debugging approaches with success rates
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

### Team Mode (Chroma Cloud -- recommended)

```bash
# Set credentials once (add to your shell profile)
export CMM_CHROMA_API_KEY="ck-..."
export CMM_CHROMA_TENANT="<tenant-uuid>"
export CMM_CHROMA_DATABASE="cmm"

# Developer 1: Initialize with Chroma Cloud shared store
cmm init . --cloud-tenant "$CMM_CHROMA_TENANT" \
           --cloud-database "$CMM_CHROMA_DATABASE" \
           --developer alice

# Push new local memories to shared staging
cmm push --project my-project

# Reviewer: approve or reject staged memories
cmm review --project my-project

# Developer 2: Same init, then pull approved memories
cmm init . --cloud-tenant "$CMM_CHROMA_TENANT" \
           --cloud-database "$CMM_CHROMA_DATABASE" \
           --developer bob
cmm pull --project my-project

# Check sync status
cmm status
```

The shared store (hosted on Chroma Cloud) has a staging area with
human-in-the-loop review: pushed nodes must be approved before they
become team-visible.

The Chroma Cloud API key is **never stored in config files** -- only
read from the `CMM_CHROMA_API_KEY` environment variable.

### Team Mode (filesystem -- self-hosted alternative)

```bash
# Developer 1: Initialize with a shared filesystem path
cmm init . --shared /shared/team/memory --developer alice

# Push / review / pull work the same as cloud mode
cmm push --project my-project
cmm review --project my-project
cmm pull --project my-project
```

### Memory Scope

Every memory is classified as:

- **PROJECT** -- specific to one repo (e.g., "Alembic needs models in `__init__.py`")
- **TEAM** -- general knowledge (e.g., "Pydantic v2 deprecated `class Config`")

Team-scope memories cross project boundaries automatically on `cmm pull`.

### Evaluation

The system tracks four helpfulness signals automatically:

| Signal | What it measures |
|--------|-----------------|
| **A: Errors Resolved** | DEAD_END -> memory retrieval -> SOLUTION within 8 messages |
| **B: Pitfalls Avoided** | Surfaced pitfall has no matching DEAD_END (embedding cosine >= 0.70) |
| **C: Pivots After Retrieval** | PIVOT within 5 messages of a /search-memory invocation |
| **D: Harmful Memory** | Loaded memory semantically matches a subsequent DEAD_END -- memory may have misled |

Signal B uses embedding similarity (not word overlap) for accurate
matching. Signal D is a false-positive tracker -- it catches cases where
memory actively pointed the agent in the wrong direction.

Position estimation uses real JSONL message counts at invocation time
(not even-distribution guesses).

**Profile quality metrics** run automatically after every ingestion when
a profile exists (no API key needed):
- **Staleness** -- do file paths referenced in insights still exist?
- **Redundancy** -- pairwise cosine > 0.85 among profile entries?
- **Coverage** -- ratio of contributing sessions vs total ingested

### Controlled A/B Comparison

Compare two sessions or two built profiles directly:

```bash
# Compare two profiles head-to-head (no extraction needed)
uv run python cognitive-memory/scripts/controlled_comparison.py \
    --compare-profiles my-project-baseline my-project-assisted

# Compare two session transcripts with cold-tier extraction
uv run python cognitive-memory/scripts/controlled_comparison.py \
    --project my-project \
    --prompt "Fix the failing test" \
    --baseline-session path/to/baseline.jsonl \
    --assisted-session path/to/assisted.jsonl \
    --cold

# Generate prompts for a fresh A/B test (dry run)
uv run python cognitive-memory/scripts/controlled_comparison.py \
    --project my-project \
    --prompt "Fix the failing test" \
    --dry-run
```

## CLI Reference

```
cmm init [dir]                Initialize .cognitive/ folder
    --cloud-tenant UUID       Enable Chroma Cloud shared mode (tenant ID)
    --cloud-database NAME     Chroma Cloud database (default: cmm)
    --shared PATH             Enable filesystem shared mode + auto-pull
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

## Exploring the Memory Store

Use the explorer script to inspect what's inside the ChromaDB store:

```bash
# Overview -- collections, projects, node types, sessions, and sample nodes
uv run python cognitive-memory/scripts/explore_store.py

# Point to a specific store directory
uv run python cognitive-memory/scripts/explore_store.py --store-dir cognitive-memory/data/memory_store

# Filter by project
uv run python cognitive-memory/scripts/explore_store.py --project mcp-gateway-registry

# Filter by node type (hypothesis, investigation, discovery, pivot, solution, dead_end, context_load)
uv run python cognitive-memory/scripts/explore_store.py --type dead_end

# Combine filters -- show only pivot nodes for a project, up to 20 results
uv run python cognitive-memory/scripts/explore_store.py --project mcp-gateway-registry --type pivot --limit 20

# Look up a single node by ID (shows full metadata, document, and embedding info)
uv run python cognitive-memory/scripts/explore_store.py --node-id node-000-01

# Semantic search over stored reasoning nodes
uv run python cognitive-memory/scripts/explore_store.py --search "debugging pydantic"

# Semantic search scoped to a project
uv run python cognitive-memory/scripts/explore_store.py --search "migration failure" --project mcp-gateway-registry

# Skip the overview tables, jump straight to node listing
uv run python cognitive-memory/scripts/explore_store.py --nodes-only --limit 20
```

The default store location is `cognitive-memory/data/memory_store`. Override
with `--store-dir` to point at any ChromaDB persistent directory.

## Running Tests

```bash
# All tests (no API keys needed -- embeddings are mocked)
uv run pytest cognitive-memory/tests/ -q

# With coverage
uv run pytest cognitive-memory/tests/ --cov=cognitive-memory/src/ --cov-report=term-missing

# A specific test file
uv run pytest cognitive-memory/tests/test_token_windowing.py -v
```

The test suite (193 tests) mocks all external API calls. No OpenAI or
Anthropic keys are needed to run tests.

## Project Structure

```
cmm/
├── pyproject.toml                # Project config (at repo root)
├── README.md
├── LICENSE
├── cognitive-memory/
│   ├── src/
│   │   ├── schemas/              # Pydantic models (session, reasoning, memory)
│   │   ├── ingestion/            # JSONL parsing + session watcher
│   │   ├── extraction/           # Warm (heuristic) + cold (LLM) extraction
│   │   ├── compression/          # Semantic dedup + profile building
│   │   ├── store/                # ChromaDB vector store (local + shared)
│   │   ├── delivery/             # MCP server + CLI query interface
│   │   ├── discovery/            # .cognitive/ folder + hooks + llms.txt
│   │   ├── evaluation/           # Session analysis, interaction logging, profile quality
│   │   ├── sync/                 # Push/pull/review for shared mode
│   │   ├── llm_client.py         # LiteLLM wrapper (Anthropic / Amazon Bedrock)
│   │   └── cli.py                # Click CLI entry point
│   ├── scripts/
│   │   ├── demo.py               # End-to-end demo (--parse-only, --clean)
│   │   ├── ingest.py             # Batch ingestion
│   │   ├── batch_consolidate.py  # Cold-tier consolidation
│   │   ├── controlled_comparison.py  # A/B comparison (--cold, --compare-profiles)
│   │   ├── explore_store.py       # ChromaDB store explorer
│   │   ├── eval_report.py        # Evaluation dashboard
│   │   └── visualize_dag.py      # DAG visualization generator
│   ├── fixtures/                 # 10 sample session transcripts
│   ├── tests/                    # 193 tests
│   ├── ARCHITECTURE.md           # Comprehensive architecture documentation
│   └── .env.example              # Environment variable template
```

## Architecture

See [ARCHITECTURE.md](cognitive-memory/ARCHITECTURE.md) for the full system
documentation including data flow diagrams, module dependency graph, schema
details, and configuration reference.

## License

MIT-0 (MIT No Attribution)
