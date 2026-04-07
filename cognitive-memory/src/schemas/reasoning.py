from pydantic import BaseModel
from enum import Enum


class NodeType(str, Enum):
    HYPOTHESIS = "hypothesis"           # agent forms a theory
    INVESTIGATION = "investigation"     # agent examines evidence
    DISCOVERY = "discovery"             # agent finds something unexpected
    PIVOT = "pivot"                     # agent changes direction
    SOLUTION = "solution"               # agent reaches a resolution
    DEAD_END = "dead_end"               # path that didn't work
    CONTEXT_LOAD = "context_load"       # agent reads/understands code


class MemoryScope(str, Enum):
    """Whether a memory is project-specific or applies team-wide."""
    PROJECT = "project"     # specific to one repo's architecture/config
    TEAM = "team"           # general knowledge applicable across repos


class ReasoningNode(BaseModel):
    """Single node in the reasoning DAG."""
    node_id: str
    node_type: NodeType
    summary: str                        # 1-2 sentence description
    evidence: str                       # what triggered this node
    message_range: tuple[int, int]      # indices into session messages
    confidence: float = 0.0            # how certain the extraction is
    scope: MemoryScope = MemoryScope.PROJECT  # default: project-specific


class ReasoningEdge(BaseModel):
    """Directed edge between reasoning nodes."""
    source_id: str
    target_id: str
    relationship: str                   # "led_to", "contradicted", "refined", etc.


class ReasoningDAG(BaseModel):
    """Directed acyclic graph of an agent's reasoning trajectory."""
    session_id: str
    nodes: list[ReasoningNode]
    edges: list[ReasoningEdge]
    pivot_nodes: list[str] = []        # node_ids where direction changed
    noise_ratio: float = 0.0          # fraction of session filtered out
