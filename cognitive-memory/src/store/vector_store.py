"""Vector store backed by ChromaDB with OpenAI text-embedding-3-small embeddings.

Supports three operating modes:

  LOCAL mode (default, backwards-compatible):
      MemoryStore(local_path="data/memory_store")
      All reads/writes go to one ChromaDB directory.

  SHARED mode — filesystem (Phase 3):
      MemoryStore(local_path="data/memory_store",
                  shared_path="/team/shared/memory_store")
      A second local ChromaDB instance acts as the team-wide shared store.

  SHARED mode — Chroma Cloud:
      MemoryStore(local_path="data/memory_store",
                  cloud_tenant="b8446dcc-...",
                  cloud_database="cmm")
      The shared store is hosted on Chroma Cloud.  The API key is read
      from the ``cloud_api_key`` parameter or the ``CMM_CHROMA_API_KEY``
      environment variable.

The shared store has TWO collections per node type: a "main" collection
(approved nodes, served to retrievals) and a "staging" collection
(pending review).

The legacy ``persist_dir=`` kwarg still works and is treated as ``local_path``.
"""
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
_COLLECTION_STAGING = "reasoning_nodes_staging"  # shared-store only
_EMBED_BATCH_SIZE = 100  # OpenAI supports up to 2048 inputs per request


class MemoryStore:
    """Persistent vector store for reasoning nodes and cognitive profiles.

    In LOCAL mode (shared_path=None): single ChromaDB instance.
    In SHARED mode: a second ChromaDB instance is opened at shared_path
    with main + staging collections. Reads can be merged across both;
    writes always go to local first and only reach shared via cmm push.
    """

    def __init__(
        self,
        local_path: str | Path | None = None,
        shared_path: str | Path | None = None,
        # Chroma Cloud shared store (takes priority over shared_path)
        cloud_api_key: str | None = None,
        cloud_tenant: str | None = None,
        cloud_database: str | None = None,
        embedding_model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        # Backwards-compat: old call sites use persist_dir=...
        persist_dir: str | Path | None = None,
    ):
        # Resolve local path: prefer new kwarg, fall back to legacy persist_dir
        if local_path is None and persist_dir is not None:
            local_path = persist_dir
        if local_path is None:
            raise ValueError("MemoryStore requires local_path (or persist_dir) to be set")

        self._local_path = str(local_path)
        self._shared_path = str(shared_path) if shared_path else None
        Path(local_path).mkdir(parents=True, exist_ok=True)

        # ── Local client (always present) ──────────────────────────
        self.client_local = chromadb.PersistentClient(
            path=self._local_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.nodes_col_local = self.client_local.get_or_create_collection(
            name=_COLLECTION_NODES,
            metadata={"hnsw:space": "cosine"},
        )
        self.profiles_col_local = self.client_local.get_or_create_collection(
            name=_COLLECTION_PROFILES,
            metadata={"hnsw:space": "cosine"},
        )

        # ── Shared client (optional) ───────────────────────────────
        self.client_shared: Any = None
        self.nodes_col_shared: Any = None
        self.staging_col_shared: Any = None
        self.profiles_col_shared: Any = None
        self._cloud_mode: bool = False

        # Cloud takes priority over filesystem shared path
        _cloud_tenant = cloud_tenant or os.environ.get("CMM_CHROMA_TENANT")
        _cloud_database = cloud_database or os.environ.get("CMM_CHROMA_DATABASE")
        _cloud_api_key = cloud_api_key or os.environ.get("CMM_CHROMA_API_KEY")

        if _cloud_tenant and _cloud_database:
            if not _cloud_api_key:
                raise ValueError(
                    "Chroma Cloud mode requires cloud_api_key or "
                    "CMM_CHROMA_API_KEY environment variable"
                )
            self._cloud_mode = True
            self.client_shared = chromadb.CloudClient(
                api_key=_cloud_api_key,
                tenant=_cloud_tenant,
                database=_cloud_database,
            )
        elif self._shared_path:
            Path(self._shared_path).mkdir(parents=True, exist_ok=True)
            self.client_shared = chromadb.PersistentClient(
                path=self._shared_path,
                settings=Settings(anonymized_telemetry=False),
            )

        if self.client_shared is not None:
            self.nodes_col_shared = self.client_shared.get_or_create_collection(
                name=_COLLECTION_NODES,
                metadata={"hnsw:space": "cosine"},
            )
            self.staging_col_shared = self.client_shared.get_or_create_collection(
                name=_COLLECTION_STAGING,
                metadata={"hnsw:space": "cosine"},
            )
            self.profiles_col_shared = self.client_shared.get_or_create_collection(
                name=_COLLECTION_PROFILES,
                metadata={"hnsw:space": "cosine"},
            )

            # Hard check: shared store must use the same embedding model
            self._verify_embedding_model_compatibility(embedding_model)

        # ── Backwards-compat aliases ───────────────────────────────
        # Existing code uses self.client / self.nodes_col / self.profiles_col
        self.client = self.client_local
        self.nodes_col = self.nodes_col_local
        self.profiles_col = self.profiles_col_local
        self._persist_dir = self._local_path  # legacy field used by batch_consolidate

        self._embedding_model = embedding_model
        self._openai = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # ── Shared-mode introspection ──────────────────────────────────────────

    @property
    def has_shared(self) -> bool:
        return self.client_shared is not None

    def _verify_embedding_model_compatibility(self, local_model: str) -> None:
        """Ensure the shared store's embedding model matches local.

        Stored as a metadata document in the shared store under a sentinel
        ID. If the shared store has never been initialized, write the
        current model. If it has, fail if it doesn't match.
        """
        if not self.client_shared:
            return
        try:
            meta_col = self.client_shared.get_or_create_collection(
                name="cmm_meta",
                metadata={"hnsw:space": "cosine"},
            )
            existing = meta_col.get(ids=["embedding_model"], include=["documents"])
            if existing["documents"]:
                stored_model = existing["documents"][0]
                if stored_model != local_model:
                    raise RuntimeError(
                        f"Shared store embedding model mismatch: "
                        f"shared='{stored_model}' but local='{local_model}'. "
                        f"All clients must use the same embedding model."
                    )
            else:
                # First time — record the model
                meta_col.upsert(
                    ids=["embedding_model"],
                    documents=[local_model],
                    metadatas=[{"set_at": datetime.now(timezone.utc).isoformat()}],
                    embeddings=[[0.0]],  # placeholder, never queried
                )
        except RuntimeError:
            raise
        except Exception:
            # Don't fail on transient errors — log silently
            pass

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

    # ── Shared-store operations (Phase 3-5) ────────────────────────────────

    def get_collection(self, source: str = "local"):
        """Pick a collection by source: 'local', 'shared', 'staging'.

        Used by sync and review code paths so they can iterate over the
        right physical collection without knowing the internal field names.
        """
        if source == "local":
            return self.nodes_col_local
        if source == "shared":
            if not self.has_shared:
                raise RuntimeError("No shared store configured")
            return self.nodes_col_shared
        if source == "staging":
            if not self.has_shared:
                raise RuntimeError("No shared store configured")
            return self.staging_col_shared
        raise ValueError(f"Unknown collection source: {source}")

    def get_unpushed_nodes(self, project_id: str) -> list[dict[str, Any]]:
        """Return local nodes that have not yet been pushed to shared.

        ChromaDB's where filter doesn't support empty-string equality
        cleanly, so we filter post-fetch.
        """
        results = self.nodes_col_local.get(
            where={"project_id": project_id},
            include=["documents", "metadatas", "embeddings"],
        )
        unpushed = []
        for nid, doc, meta, emb in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["embeddings"],
        ):
            if not meta.get("pushed_at"):
                unpushed.append({
                    "id": nid,
                    "summary": doc,
                    "embedding": emb,
                    **meta,
                })
        return unpushed

    def mark_pushed(self, ids: list[str], when: str | None = None) -> int:
        """Stamp local nodes as pushed. Idempotent."""
        if not ids:
            return 0
        when = when or datetime.now(timezone.utc).isoformat()
        existing = self.nodes_col_local.get(ids=ids, include=["metadatas"])
        new_metas = []
        for meta in existing["metadatas"]:
            updated = dict(meta)
            updated["pushed_at"] = when
            new_metas.append(updated)
        self.nodes_col_local.update(ids=existing["ids"], metadatas=new_metas)
        return len(existing["ids"])

    def stage_to_shared(
        self,
        nodes: list[dict[str, Any]],
        developer: str = "",
    ) -> int:
        """Write nodes into the shared store's staging collection.

        Each input node dict must contain: id, summary, embedding, plus
        existing metadata fields. The staging entry is marked
        approved=False until reviewed.
        """
        if not self.has_shared or not nodes:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for n in nodes:
            ids.append(n["id"])
            documents.append(n["summary"])
            embeddings.append(n["embedding"])
            meta = {k: v for k, v in n.items() if k not in {"id", "summary", "embedding"}}
            meta["approved"] = False
            meta["approved_by"] = ""
            meta["staged_at"] = datetime.now(timezone.utc).isoformat()
            if developer:
                meta["source_developer"] = developer
            metadatas.append(meta)

        self.staging_col_shared.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return len(ids)

    def list_pending_in_staging(
        self, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return all unapproved nodes from the shared staging collection."""
        if not self.has_shared:
            return []
        where = {"project_id": project_id} if project_id else None
        try:
            results = self.staging_col_shared.get(
                where=where,
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception:
            return []
        embs = results.get("embeddings")
        if embs is None:
            embs = [None] * len(results["ids"])
        out = []
        for nid, doc, meta, emb in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            embs,
        ):
            if not meta.get("approved"):
                out.append({
                    "id": nid,
                    "summary": doc,
                    "embedding": emb,
                    **meta,
                })
        return out

    def promote_from_staging(
        self,
        node_ids: list[str],
        approver: str,
        scope_overrides: dict[str, str] | None = None,
        summary_overrides: dict[str, str] | None = None,
    ) -> int:
        """Move staged nodes into the shared main collection.

        Optionally overrides the scope or summary for each node before promotion.
        """
        if not self.has_shared or not node_ids:
            return 0

        results = self.staging_col_shared.get(
            ids=node_ids,
            include=["documents", "metadatas", "embeddings"],
        )
        if not results["ids"]:
            return 0

        when = datetime.now(timezone.utc).isoformat()
        promote_ids = []
        promote_docs = []
        promote_embs = []
        promote_metas = []

        for nid, doc, meta, emb in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["embeddings"],
        ):
            new_meta = dict(meta)
            new_meta["approved"] = True
            new_meta["approved_by"] = approver
            new_meta["approved_at"] = when
            if scope_overrides and nid in scope_overrides:
                new_meta["scope"] = scope_overrides[nid]
            new_doc = summary_overrides[nid] if summary_overrides and nid in summary_overrides else doc
            promote_ids.append(nid)
            promote_docs.append(new_doc)
            promote_embs.append(emb)
            promote_metas.append(new_meta)

        self.nodes_col_shared.upsert(
            ids=promote_ids,
            embeddings=promote_embs,
            metadatas=promote_metas,
            documents=promote_docs,
        )
        # Remove from staging
        self.staging_col_shared.delete(ids=promote_ids)
        return len(promote_ids)

    def reject_in_staging(
        self,
        node_ids: list[str],
        reviewer: str,
        reason: str = "",
    ) -> int:
        """Mark staged nodes as rejected. Kept in staging for audit."""
        if not self.has_shared or not node_ids:
            return 0
        results = self.staging_col_shared.get(ids=node_ids, include=["metadatas"])
        if not results["ids"]:
            return 0
        when = datetime.now(timezone.utc).isoformat()
        new_metas = []
        for meta in results["metadatas"]:
            new = dict(meta)
            new["approved"] = False
            new["rejected"] = True
            new["rejected_by"] = reviewer
            new["rejected_at"] = when
            new["rejection_reason"] = reason
            new_metas.append(new)
        self.staging_col_shared.update(ids=results["ids"], metadatas=new_metas)
        return len(results["ids"])

    def list_approved_shared(
        self,
        project_id: str | None = None,
        include_team: bool = True,
    ) -> list[dict[str, Any]]:
        """Return approved nodes from the shared main collection.

        If include_team is True, also returns all approved team-scope nodes
        regardless of project_id.
        """
        if not self.has_shared:
            return []

        out = []
        seen_ids: set[str] = set()

        # Project-scoped approved nodes for this project
        if project_id:
            try:
                results = self.nodes_col_shared.get(
                    where={"project_id": project_id},
                    include=["documents", "metadatas", "embeddings"],
                )
                for nid, doc, meta, emb in zip(
                    results["ids"], results["documents"],
                    results["metadatas"], results["embeddings"],
                ):
                    if meta.get("approved") and nid not in seen_ids:
                        out.append({"id": nid, "summary": doc, "embedding": emb, **meta})
                        seen_ids.add(nid)
            except Exception:
                pass

        # Team-scope approved nodes (any project)
        if include_team:
            try:
                results = self.nodes_col_shared.get(
                    where={"scope": "team"},
                    include=["documents", "metadatas", "embeddings"],
                )
                for nid, doc, meta, emb in zip(
                    results["ids"], results["documents"],
                    results["metadatas"], results["embeddings"],
                ):
                    if meta.get("approved") and nid not in seen_ids:
                        out.append({"id": nid, "summary": doc, "embedding": emb, **meta})
                        seen_ids.add(nid)
            except Exception:
                pass

        return out

    def upsert_pulled_nodes(
        self,
        nodes: list[dict[str, Any]],
        project_id: str,
    ) -> int:
        """Insert nodes pulled from the shared store into the local store.

        Marks them as pushed_at=their original push time so they don't get
        re-pushed back to shared on the next push cycle.
        """
        if not nodes:
            return 0
        ids = []
        docs = []
        embs = []
        metas = []
        now = datetime.now(timezone.utc).isoformat()
        for n in nodes:
            ids.append(n["id"])
            docs.append(n["summary"])
            embs.append(n["embedding"])
            meta = {k: v for k, v in n.items() if k not in {"id", "summary", "embedding"}}
            # Force project_id to match local view (team nodes may have a different one)
            if meta.get("scope") != "team":
                meta["project_id"] = project_id
            # Always stamp pushed_at — pulled nodes came from shared, don't bounce
            # back on next push. setdefault is wrong here because pushed_at may
            # already be present as an empty string from prior writes.
            if not meta.get("pushed_at"):
                meta["pushed_at"] = now
            metas.append(meta)
        self.nodes_col_local.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
        return len(ids)
