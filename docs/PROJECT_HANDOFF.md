# Project Context Handoff — Model-Agnostic Local AI Harness

**Last verified:** 2026-09-04 against the live workspace.

**Historical detailed audit:** `docs/PROJECT_AUDIT_FULL.md`

## How to work with me

- Keep responses minimal and easy to scan.
- I have ADHD.
- Prefer bullets over long paragraphs.
- Do not re-litigate confirmed architectural decisions without a concrete reason.
- Make one logical code change at a time.
- Verify live code before assuming this handoff is still accurate.
- Only create Git commits when explicitly asked.

## 1. What the project is

- Local, single-user AI harness whose default backend loads `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4.
- Shared `HarnessRunner` used by:
  - `chat.py` — canonical CLI with history and compression.
  - `infer.py` — stateless alternate CLI.
  - `webapp.py` — FastAPI UI with persistent chat sessions and document management.
- Main subsystems:
  - Hybrid intent routing (`intent_classifier.py`).
  - Long-term user memory (`memory_core.py` + `data/agent_memory.db`).
  - TXT/PDF RAG (`documents/`, `knowledge/`, `data/documents.db`).
  - Stateless, allowlisted general-purpose tools (`tools/`).
  - Web session history (`data/chat_sessions.db`).
- `experiments/train.py` and `experiments/tool_calling_test.py` are historical experiments, not production entry points.

## 2. Current architecture

```text
chat.py / infer.py / webapp.py
              │
              ▼
        HarnessRunner
      ├── hybrid intent router
      ├── OfflineMemoryManager ──► data/agent_memory.db
      ├── DocumentIndex ─────────► data/documents.db + knowledge/
      ├── ToolSelector ──────────► optional Needle2 / main-model fallback
      ├── ToolManager ───────────► local/web utility registry
      └── ModelBackend ──────────► TransformersBackend ──► Qwen or SmolLM2

BGE-small-en-v1.5 runs on CPU and is shared by memory and documents.
Web messages are stored separately in `data/chat_sessions.db`. Runtime databases and `data/chatbot_debug.log` share the `data/` directory but remain independent files. Defaults are centralized in `app_paths.py` and can be relocated with `CHATBOT_DATA_DIR` or the per-file path overrides.
```

Intent, memory, documents, tools, and session history are intentionally separate.

## 3. Turn lifecycle

1. Deterministic intent evidence resolves high-confidence flags; Qwen classifies only unresolved flags.
2. Contextual memory commands resolve to the most recent confidently durable user assertion.
3. Memory is queried only for `memory_read`.
4. Documents are queried only for `document_read`.
5. The system prompt includes only context actually retrieved.
6. Chat/web use history; `infer.py` uses only the current turn.
7. Chat/web may summarize old history and trim lowest-ranked document chunks.
8. When `tool_use` is true, the configured `ToolSelector` receives registered schemas and previous structured results. Needle2 is used when configured; otherwise the active main model provides fallback selection through its backend protocol.
9. Each request is treated as untrusted, allowlisted, validated, checked against failed-call signatures, executed, and returned in a structured envelope; protocol messages remain ephemeral.
10. If `memory_write` is true, Qwen extracts facts from the selected evidence source and Python validates/stores them.
11. Web persists the user/final-assistant pair and returns retrieval trace metadata.

`general_chat` is descriptive metadata; it does not gate the main reply.

## 4. Tool system — production, not experimental

Production tool calling lives in `tools/` and `harness/runner.py`.

The registry includes die/random utilities, safe expression/list aggregation,
Frankfurter daily currency reference rates/conversion, structured Tavily web
search, bounded web text fetching, Open-Meteo weather, one-off local file
reading, declarative CSV/XLSX analysis, and bounded directory listing.

Important behavior:

- Only registry tools execute; unknown names return a structured error.
- Arguments must be a JSON object and pass type/range/unexpected-field validation.
- Calculator uses a restricted AST evaluator; production code does not use `eval()`.
- Tool use is opt-in through the independent `tool_use` intent flag; the default budget is eight attempted calls per turn.
- Calls execute sequentially, including repeated tools; exhaustion forces a final tools-disabled answer.
- An identical call that already failed in the current turn is not executed again. The selector receives `duplicate_failed_call`; changed arguments and repeated successful calls remain allowed.
- Requested action counts are tracked per turn; dependent calculator calls wait for actual prerequisite results, while independent heterogeneous calls may execute in one model response.
- Tool call/result messages are temporary for the follow-up generation; only the final answer enters normal chat history.
- Tools are stateless and never write to memory, document, or session databases.
- Tool logs record schema exposure, generation round, parsed-call count, validation/execution results, and budget exhaustion without adding raw model output.
- Tavily web search reads `TAVILY_API_KEY` only inside the tool. Missing
  credentials fail explicitly; search results cannot be replaced by
  `fetch_webpage` or model-authored prose.
- The old experiment still uses `eval()` but is isolated under `experiments/`.

Verification: the automated suite covers all ten registered tools,
malformed/duplicate tool-call handling, bounded multi-call orchestration, and
the three-spreadsheet pipeline. A current live-Qwen regression over every tool
is still desirable.

## 5. Memory

- User-only durable facts; arbitrary third-party graph memory is intentionally unsupported.
- Trusted relations:
  - `name`, `lives_in`, `works_in`, `codes_in`
  - `favorite_programming_language`, `recently_codes_in`
  - `prefers_beverage`, `changed_mind*`
- Exact structured lookup precedes broad tiered retrieval.
- Tier 1: five-row persistent hot cache.
- Tier 2: semantic search over active archive rows, threshold `> 0.50`, top 2 unseen relations.
- Tier 3: `global_archive`, including invalidated history.
- Exact canonical duplicates are rejected; near-duplicate prose threshold is `0.96` within the same relation.
- Changed values invalidate old rows rather than deleting them.
- Programming usage, favorite/preference, and recent/current activity are separate relations.
- Web `ReadOnlyOfflineMemoryManager` suppresses retrieval cache mutation only. Web memory writes still persist through inherited write methods.

Live snapshot at handoff:

- 6 archive rows, 4 active, 4 cached.
- One active canonical row each for name, location, favorite programming language, and beverage preference.
- The old duplicate-active-row defect is fixed in code and no longer present in the active live data.

## 6. Documents

- Sources: recursive `.txt` and `.pdf` discovery under `knowledge/`.
- Extractors: text decoding and `pypdf`; no OCR.
- Chunking: target 350 tokens, overlap 60, sentence-aware.
- Router threshold `0.22`, keyword boost `0.08`.
- Retrieval threshold `0.55`, top 4 chunks.
- Explicit filenames scope retrieval; missing named files do not inject semantic decoys.
- Sync runs at startup and after web upload/delete.
- Web upload sanitizes basenames, accepts only `.txt`/`.pdf`, enforces a 10 MB backend limit, rejects collisions, and never overwrites an existing document.

Live snapshot: 9 indexed documents, 9 chunks, 5 TXT + 4 PDF files, no deleted DB row.

## 7. Model and entry-point configuration

- Default main model: `Qwen/Qwen2.5-3B-Instruct`; `HuggingFaceTB/SmolLM2-1.7B-Instruct` is selectable at startup or from the web UI.
- `ModelSpec` and `ModelCapabilities` declare model identity, backend, supported behavior, and load options; registry entries contain no loader callables.
- `create_backend()` constructs the backend. Entry points call `load()`, and normal CLI exit, web switching, and web shutdown call `close()`.
- `HarnessRunner` sends `GenerationRequest` and consumes `GenerationResult`; it does not access tokenizers, tensors, devices, quantization, or model-family syntax.
- `TransformersBackend` owns loading, tokenization, chat-template rendering, generation, decoding, device placement, and quantization.
- Qwen uses its native tool-aware chat template and 4-bit bitsandbytes NF4 with BF16 compute, double quantization, and automatic device placement.
- SmolLM2 remains non-quantized and uses a compact typed-signature `<tool_call>`/`<tool_result>` compatibility policy inside the backend. Its system instructions are coalesced before actionable messages so late harness enforcement does not displace the user request.
- Tool selector: main-model fallback by default; optional native `Cactus-Compute/needle2` through `CHATBOT_TOOL_SELECTOR=needle2` and `requirements-needle.txt`.
- Main-model switching does not reload or replace the active tool selector.
- Embeddings: `BAAI/bge-small-en-v1.5`, CPU, float32 SQLite blobs.
- `chat.py`: history on; compression threshold 1,500; keep 2 turns; 300 reply tokens.
- `infer.py`: history/compression off; 450 reply tokens.
- `webapp.py`: history/compression on; default 1,500/2; 450 reply tokens.
- Reply sampling: temperature 0.7, top-p 0.9.
- Memory extraction: deterministic, max 120 tokens.
- Intent classification: deterministic, default max 72 tokens.
- Summary generation: deterministic, default max 200 tokens.
- Tool schemas are passed only when `tool_use` is true.

Harness generation settings, capability switches, and maximum tool steps are model-independent. The 1,500-token threshold is not a hard cap; oversized current input, memory, or retained history can still exceed it.

## 8. Web application

- FastAPI/uvicorn on `127.0.0.1:8000`.
- Model/index load in a background initialization thread.
- One global generation lock serializes inference.
- Cookie `roots_chat_session`, HTTP-only, `SameSite=Lax`.
- Session create/list/read/activate/delete endpoints.
- Document list/upload/delete endpoints.
- UI displays per-turn memory and document trace metadata.
- Chat payload length: 1–20,000 characters.

Current bug: `/reset` clears in-memory messages and resets the turn counter but preserves DB messages. Reload can restore old history and produce overlapping turn numbers.

Live session snapshot: 1 session, 18 messages, WAL mode active.

## 9. Logging and data sensitivity

- `data/chatbot_debug.log` uses a 5 MB rotating handler with three backups.
- Clearing logger handlers prevents duplicates; it does not truncate existing files.
- Windows attempts to launch a PowerShell live tail.
- Prompt assembly and assistant replies log character counts rather than content. Tool payloads remain bounded and redact secret-like fields and URL query values.

Treat the log as sensitive local data.

## 10. Tests

Last command run:

```powershell
.\qwen-env\Scripts\python.exe -m unittest discover -s tests -v
```

Results on 2026-09-04 after the backend-independent model migration:
**196 pytest tests passed (plus 70 subtests)**. The earlier unittest discovery
run also passed **80/80**; pytest remains the canonical full suite.

Covered:

- Backend contracts, declarative registry/factory, lifecycle, generation requests/results, Qwen NF4 configuration, native Qwen tool templates, and SmolLM2 textual tool normalization.
- Hybrid intent parsing/routing/repair/fallback/context inheritance.
- Memory gating, grounding, duplicates, temporal updates, relation families, and history queries.
- Prompt conditioning, compression rules, and document-budget trimming.
- Filename-scoped RAG and missing-file decoy prevention.
- Tool schemas, parsing, validation, safe calculator, execution, failure responses, and orchestration.
- Frankfurter daily reference-rate lookup/conversion, provider failures,
  response bounds, currency routing, and result grounding.
- Tavily search success/failures, credential isolation, response/result bounds,
  search routing, required execution, and rejection of fabricated/scraper
  fallback completion.

Still missing or not rerun for this revision:

- Current live-Qwen tool smoke for all registered tools and non-tool prompts.
- Broader live-SmolLM2 smoke across every registered tool (a real calculator production-path smoke passes).
- Full CLI/web startup and restart test.
- Concurrent web stress.
- Corrupt/scanned PDF behavior.
- DB recovery.
- Very long input and true model-context-limit behavior.

## 11. Known risks, in priority order

1. Current real-model tool selection/argument quality lacks an automated regression for both configured main models.
2. Web reset semantics can desynchronize RAM and persisted history.
3. The prompt budget is not a hard cap.
4. Retrieval is brute-force and document routing/retrieval can encode the query twice.
5. “ReadOnly” web memory suppresses retrieval-cache writes but can still persist intentionally routed fact writes.
6. No auth, CSRF protection, or rate limiting; keep the server localhost-only.
7. Intermittent historical native Qwen startup exits remain unexplained, although the current environment has loaded successfully.

## 12. Locked or intentional decisions

- Separate SQLite databases for memory, documents, and web sessions.
- Document source files in `knowledge/`; RAG code in `documents/`.
- BGE on CPU to preserve GPU memory for Qwen.
- One shared `HarnessRunner` for all entry points; `ConversationOrchestrator` remains a compatibility alias in `orchestrator.py`.
- Deterministic routing evidence has precedence over Qwen classification.
- User-only durable memory with temporal invalidation.
- 4-bit NF4 inference.
- Tool execution is stateless and restricted to a small allowlisted registry; model-generated names and arguments are untrusted.

## 13. Recommended next work

1. Add opt-in real-Qwen and real-SmolLM2 generation/tool smoke coverage through `ModelBackend`.
2. Fix or redefine web reset persistence semantics.
3. Enforce a true end-to-end prompt/context limit.
4. Measure and improve retrieval scaling and duplicate query embedding work.
5. Validate clean-environment installation from the pinned dependency metadata.

## 14. Rules for future agents

- Treat live code and live DB state as the source of truth.
- Read relevant files before editing; this handoff can drift.
- Preserve separation among memory, documents, sessions, intent, and tools unless explicitly redesigning it.
- Keep model execution behind `models.ModelBackend`; Transformers loading, quantization, and model-specific prompt policies belong in `models/backends.py`.
- Keep tool execution allowlisted and stateless; treat every model-produced tool name and argument as untrusted.
- Add tools only through the registry plus centralized validation and tests.
- Do not use the experiment’s `eval()` calculator in production.
- Do not describe tool calling as experimental-only; the current production harness uses it.
- Do not claim there are no automated tests; the current suite has 196 pytest tests (plus 70 subtests), and the latest unittest discovery run has 80 tests.
- `chat.py` is canonical unless the user says otherwise.
- Do not create commits unless explicitly requested.
