"""Vector store backed by ChromaDB with OpenAI text-embedding-3-small embeddings."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from ..schemas.memory import CognitiveProfile
from ..schemas.reasoning import ReasoningDAG, ReasoningNode

_DEFAULT_MODEL = "text-embedding-3-small"
_COLLECTION_NODES = "reasoning_nodes"
_COLLECTION_PROFILES = "cognitive_profiles"
_EMBED_BATCH_SIZE = 100  # OpenAI supports up to 2048 inputs per request


class MemoryStore:
    """Persistent vector store for reasoning nodes and cognitive profiles."""

    def __init__(
        self,
        persist_dir: str | Path,
        embedding_model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        self._persist_dir = str(persist_dir)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.nodes_col = self.client.get_or_create_collection(
            name=_COLLECTION_NODES,
            metadata={"hnsw:space": "cosine"},
        )
        self.profiles_col = self.client.get_or_create_collection(
            name=_COLLECTION_PROFILES,
            metadata={"hnsw:space": "cosine"},
        )

        self._embedding_model = embedding_model
        self._openai = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # ── Embedding ──────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI text-embedding-3-small."""
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            response = self._openai.embeddings.create(
                model=self._embedding_model,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings

    # ── Store DAG nodes ────────────────────────────────────────────────────

    def store_dag(self, dag: ReasoningDAG, project_id: str) -> int:
        """Embed and store all nodes from a DAG. Returns number stored."""
        if not dag.nodes:
            return 0

        texts = [f"{n.node_type.value}: {n.summary}" for n in dag.nodes]
        embeddings = self.embed(texts)

        ids = [f"{project_id}::{dag.session_id}::{n.node_id}" for n in dag.nodes]
        metadatas = [
            {
                "project_id": project_id,
                "session_id": dag.session_id,
                "node_id": n.node_id,
                "node_type": n.node_type.value,
                "confidence": n.confidence,
                "msg_start": n.message_range[0],
                "msg_end": n.message_range[1],
                "is_pivot": n.node_id in dag.pivot_nodes,
                # Sync / scope metadata (Phase 2-5)
                "scope": n.scope.value,
                "pushed_at": "",          # empty until cmm push runs
                "approved": False,         # only meaningful in shared collections
                "approved_by": "",
                "source_developer": os.environ.get("CMM_DEVELOPER_NAME", ""),
            }
            for n in dag.nodes
        ]
        documents = [n.summary for n in dag.nodes]

        self.nodes_col.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return len(dag.nodes)

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        project_id: str | None = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over stored reasoning nodes.

        Args:
            scope: optional filter — "project", "team", or None (both).
        """
        where_clauses: list[dict] = []
        if project_id:
            where_clauses.append({"project_id": project_id})
        if scope:
            where_clauses.append({"scope": scope})

        if len(where_clauses) > 1:
            where = {"$and": where_clauses}
        elif where_clauses:
            where = where_clauses[0]
        else:
            where = None

        embedding = self.embed([query])[0]
        results = self.nodes_col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.nodes_col.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if meta.get("confidence", 1.0) >= min_confidence:
                hits.append(
                    {
                        "summary": doc,
                        "similarity": 1.0 - dist,  # cosine distance → similarity
                        **meta,
                    }
                )
        return hits

    # ── Get all nodes for a project ────────────────────────────────────────

    def get_all_nodes(self, project_id: str) -> list[dict[str, Any]]:
        """Return all stored nodes for a project."""
        count = self.nodes_col.count()
        if count == 0:
            return []
        results = self.nodes_col.get(
            where={"project_id": project_id},
            include=["documents", "metadatas", "embeddings"],
        )
        nodes = []
        for doc, meta, emb in zip(
            results["documents"],
            results["metadatas"],
            results["embeddings"],
        ):
            nodes.append({"summary": doc, "embedding": emb, **meta})
        return nodes

    def get_embeddings_for_nodes(self, project_id: str) -> tuple[list[str], list[list[float]]]:
        """Return (ids, embeddings) for all nodes in a project."""
        count = self.nodes_col.count()
        if count == 0:
            return [], []
        results = self.nodes_col.get(
            where={"project_id": project_id},
            include=["embeddings"],
        )
        return results["ids"], results["embeddings"]

    # ── Cognitive profiles ─────────────────────────────────────────────────

    def save_profile(self, profile: CognitiveProfile) -> None:
        """Persist a cognitive profile."""
        text = f"Cognitive profile for {profile.project_id}: " + " ".join(
            [i.insight for i in profile.architectural_insights]
            + [p.description for p in profile.pitfalls]
        )
        embedding = self.embed([text])[0]

        self.profiles_col.upsert(
            ids=[profile.project_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "project_id": profile.project_id,
                    "session_count": profile.session_count,
                    "last_updated": profile.last_updated.isoformat(),
                }
            ],
            documents=[profile.model_dump_json()],
        )

    def get_profile(self, project_id: str) -> CognitiveProfile | None:
        """Retrieve a stored cognitive profile."""
        try:
            results = self.profiles_col.get(
                ids=[project_id],
                include=["documents"],
            )
            if results["documents"]:
                return CognitiveProfile.model_validate_json(results["documents"][0])
        except Exception:
            pass
        return None

    # ── Diagnostics ────────────────────────────────────────────────────────

    def node_count(self, project_id: str | None = None) -> int:
        if project_id is None:
            return self.nodes_col.count()
        results = self.nodes_col.get(
            where={"project_id": project_id},
            include=[],
        )
        return len(results["ids"])

    def list_projects(self) -> list[str]:
        count = self.nodes_col.count()
        if count == 0:
            return []
        results = self.nodes_col.get(include=["metadatas"])
        seen: set[str] = set()
        for meta in results["metadatas"]:
            pid = meta.get("project_id")
            if pid:
                seen.add(pid)
        return sorted(seen)
