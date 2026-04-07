from pydantic import BaseModel, Field
from datetime import datetime

from .reasoning import MemoryScope


class Pitfall(BaseModel):
    description: str
    frequency: int = 1                          # how many sessions encountered this
    severity: str = "medium"                    # low, medium, high
    resolution_strategy: str | None = None
    scope: MemoryScope = MemoryScope.PROJECT


class DiagnosticStrategy(BaseModel):
    trigger: str                                # when to use this strategy
    steps: list[str]                            # ordered diagnostic steps
    success_rate: float = 0.0                  # how often this works
    source_sessions: list[str] = []            # which sessions derived this
    scope: MemoryScope = MemoryScope.PROJECT


class ArchitecturalInsight(BaseModel):
    component: str                              # which part of the codebase
    insight: str                                # what the agent learned
    confidence: float = 0.0
    scope: MemoryScope = MemoryScope.PROJECT


class CognitiveProfile(BaseModel):
    """The consolidated output: what an agent should know about this codebase."""
    project_id: str
    last_updated: datetime
    architectural_insights: list[ArchitecturalInsight] = []
    pitfalls: list[Pitfall] = []
    diagnostic_strategies: list[DiagnosticStrategy] = []
    key_patterns: list[str] = []               # recurring patterns observed
    anti_patterns: list[str] = []              # things that consistently fail
    session_count: int = 0                     # how many sessions contributed
