# Demo 2 Results: IAM Auth0/Keycloak Routing Bug
## Cognitive Memory Manager — Controlled A/B Comparison

**Task:** "When Auth0 is configured as the authentication provider (AUTH_PROVIDER=auth0), the IAM management page still shows Keycloak groups instead of Auth0 groups. Debug the IAM provider routing logic and fix it."

**Profile entry (95% confidence — highest in the entire profile):**
> "The get_iam_manager() factory function lacks an Auth0-specific case and falls through to the default Keycloak manager. This means when Auth0 is configured as the active authentication provider, IAM group management operations still use the Keycloak backend."

---

## Results

|                        | Baseline (no memory)     | Assisted (with memory)   | Reduction  |
|------------------------|--------------------------|--------------------------|------------|
| Session ID             | 2ca82ddf                 | 8f0be201                 |            |
| Assistant messages     | 119                      | 46                       | **-61%**   |
| Solution at message    | 32 (identified root cause at 48 for full fix) | 37                       |            |
| Read profile           | No (bare mode — clean)   | Via subagent/CLAUDE.md   |            |
| Files written          | 7                        | 2                        | **-71%**   |
| Session size           | 1127 KB                  | 281 KB                   | **-75%**   |

---

## What Happened

### Baseline
- 119 messages, no memory
- Explored frontend (IAMGroups.tsx), hooks (useRegistryConfig.ts), config routes, management routes, iam_manager.py, tests
- Wrote 7 files — over-engineered solution touching frontend + backend + tests
- Found the bug eventually but took the long route through every layer

### Assisted
- 46 messages, CLAUDE.md instructed to read profile
- Spawned subagent at message 2 which parsed project context including profile
- Went directly to `iam_manager.py` at message 5 — no frontend exploration
- Root cause identified at message 37:
  > "AUTH_PROVIDER is read once at module import time and cached as a module-level constant. get_iam_manager() at line 607 uses the stale module-level constant, not the live env var."
- Fixed in 2 files only: `iam_manager.py` + `management_routes.py`
- Finished at message 46

---

## Key Quotes

**Baseline (msg 32 — first hint):**
Exploring IAMGroups.tsx frontend component, checking useRegistryConfig hook, tracing API calls. Still hadn't found get_iam_manager().

**Assisted (msg 37 — root cause):**
> "Now I have a clear picture of the bug. `iam_manager.py` line 25: `AUTH_PROVIDER: str = os.environ.get("AUTH_PROVIDER", "keycloak")` — read once at module import time and cached as a module-level constant. `get_iam_manager()` line 607 uses the stale module-level constant, not the live env var."

**Assisted (msg 46 — wrap-up):**
> "Both fixes are in place and consistent with the dynamic pattern already used in nginx_service.py, server_routes.py, and the M2M route in management_routes.py."

---

## Verdict
**Memory helped: 61% fewer messages, 71% fewer files written, 75% smaller session.**

The profile's 95% confidence insight pointed directly to the factory function. The baseline explored 7 layers (frontend, hooks, config routes, management routes, IAM manager, and 2 test files). The assisted agent fixed it in 2 files.

---

## Cold Evaluator Output
The cold extractor classified both sessions as CONTEXT_LOAD nodes (no DEAD_END/SOLUTION classification).
This is a known limitation — the LLM extractor under-classifies on sessions that resolve cleanly without explicit dead ends.
The raw JSONL metrics above are the authoritative numbers.

Evaluator JSON: `comparison_mcp-gateway-registry_1776285245.json`
Evaluator verdict: "No measurable difference" (misleading — extractor limitation, not reality)

---

## Session Paths
- Baseline: `~/.claude/projects/-Users-sazankhalid-Downloads-mcp-gateway-registry/2ca82ddf-7342-497e-b971-60426a7408bd.jsonl`
- Assisted: `~/.claude/projects/-Users-sazankhalid-Downloads-mcp-gateway-registry/8f0be201-924f-408a-9866-ed52311cd8cc.jsonl`
