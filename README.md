# Qwen Local Chatbot

A private, local-first assistant built around `Qwen/Qwen2.5-3B-Instruct`. It
combines conversational history, durable user memory, TXT/PDF retrieval, and a
small allowlisted tool system behind CLI and FastAPI interfaces.

The model and embeddings run locally after their first download. The weather,
daily reference-rate, Tavily web-search, and webpage tools are the only production
features that intentionally make outbound requests.

## Features

- 4-bit NF4 Qwen inference with automatic device placement.
- Stateful CLI and browser UI, plus a stateless CLI variant.
- SQLite-backed user memory and web chat sessions.
- Recursive TXT/PDF indexing with sentence-aware chunking and semantic search.
- Validated tools for calculations, random values, daily currency reference
  rates, structured web search, weather, webpage text, local file reading,
  directory listing, and spreadsheet analysis.
- Fully Needle 2-powered structured intent classification and bounded context compression.
- A broad test suite that does not require loading Qwen for normal unit tests.

## Architecture

```text
chat.py / infer.py / webapp.py
              |
              v
   ConversationOrchestrator
      |-- Needle 2 intent classification
      |-- OfflineMemoryManager --> data/agent_memory.db
      |-- DocumentIndex ---------> data/documents.db + knowledge/
      |-- ToolManager -----------> allowlisted local/web tools
      `-- Qwen2.5-3B-Instruct
```

Runtime databases, logs, model caches, and virtual environments are deliberately
excluded from Git. See the [Needle2 benchmark plan](docs/NEEDLE2_BENCHMARK_PLAN.md)
for this branch's evaluation scope and the [issue ledger](docs/PROJECT_ISSUES.md)
for verified changes. The older handoff and audit remain historical context.

## Requirements

- Python 3.11 or newer.
- An NVIDIA GPU and a working CUDA-enabled PyTorch installation are strongly
  recommended for the configured 4-bit model.
- Enough disk space for Qwen, Needle 2, and `BAAI/bge-small-en-v1.5` in the Hugging Face
  cache.

`bitsandbytes` and CUDA support vary by operating system and hardware. Install
the PyTorch build appropriate for your CUDA environment if the generic
dependency installation does not select it correctly.

## Setup

From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first application start downloads the model and embedding weights from
Hugging Face. No API key is required for public models, although an authenticated
Hugging Face account can provide more reliable download limits.

## Run

Start the canonical, history-aware CLI:

```powershell
python chat.py
```

Start the browser application at <http://127.0.0.1:8000>:

```powershell
python webapp.py
```

Run a stateless CLI turn loop:

```powershell
python infer.py
```

Enter `exit` to leave a CLI session. Keep the web server bound to localhost:
it has no authentication, CSRF protection, or rate limiting.

## Documents and local data

Place `.txt` and `.pdf` files anywhere under `knowledge/`. The index synchronizes
at startup; the web UI can also upload and remove documents. Scanned PDFs need
OCR before this project can extract their text.

By default, mutable state is written under `data/`:

- `agent_memory.db` stores durable user facts.
- `documents.db` stores the retrieval index.
- `chat_sessions.db` stores browser sessions.
- `chatbot_debug.log` contains diagnostic output and may include sensitive text.

Back up or delete this directory independently of the source tree. It is not
committed.

## Configuration

Storage and document settings can be overridden with environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CHATBOT_DATA_DIR` | Runtime state directory | `./data` |
| `CHATBOT_MEMORY_DB` | Memory database path | `<data>/agent_memory.db` |
| `CHATBOT_SESSIONS_DB` | Web session database path | `<data>/chat_sessions.db` |
| `CHATBOT_DOCUMENT_DB` | Document index path | `<data>/documents.db` |
| `CHATBOT_DEBUG_LOG` | Debug log path | `<data>/chatbot_debug.log` |
| `CHATBOT_DOCS_DIR` | Source document directory | `./knowledge` |
| `CHATBOT_DOC_CHUNK_SIZE` | Approximate chunk tokens | `350` |
| `CHATBOT_DOC_CHUNK_OVERLAP` | Approximate overlap tokens | `60` |
| `CHATBOT_DOC_TOP_K` | Maximum retrieved chunks | `4` |
| `TOOLS_ALLOWED_ROOTS` | Local-tool roots, separated by the OS path separator | Project root |

Tool safety and size limits use `TOOLS_*` variables documented alongside their
defaults in [`tools/config.py`](tools/config.py). Private-network webpage access
is disabled unless `TOOLS_ALLOW_PRIVATE_WEB_HOSTS=true` is explicitly set.
Structured web search is disabled unless `TAVILY_API_KEY` is present in the
process environment. The key is sent only to Tavily's fixed search endpoint in
the authorization header and is never a model-supplied tool argument. For a
PowerShell session, configure it before starting the application:

```powershell
$env:TAVILY_API_KEY = "<your-tavily-api-key>"
```

## Tests

Run the fast automated suite without loading the Qwen model:

```powershell
python -m pytest
```

The deterministic suite was last verified on 2026-09-04: **143 tests and 57
subtests passed**. Live Qwen generation and external API calls are not part of
this command.

The real-model routing smoke test is intentionally outside normal discovery
because it is GPU- and download-intensive:

```powershell
python tests/real_qwen_routing_smoke.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing orchestration, persistence,
or the executable tool registry.

## Repository map

- `chat.py` - canonical CLI.
- `webapp.py` - FastAPI server and embedded browser UI.
- `infer.py` - stateless CLI variant.
- `orchestrator.py` - shared turn pipeline and tool loop.
- `intent_classifier.py` - schema-constrained Needle 2 intent decisions.
- `memory_core.py` - durable fact extraction and retrieval.
- `documents/` - document discovery, extraction, indexing, and retrieval.
- `tools/` - schemas, validation, and allowlisted implementations.
- `tests/` - automated regression suite.
- `experiments/` - non-production model and training experiments.

## Current limitations

- Currency conversion uses daily central-bank reference rates, not intraday or
  tradable market prices.
- Web search requires a separately provisioned Tavily API key and returns
  result metadata/snippets, not rendered or authenticated webpage content.
- Web reset behavior needs hardening.
- Diagnostic logs contain operational metadata and bounded tool payloads; protect
  the local `data/` directory accordingly.
- Retrieval is brute-force and intended for a modest local document collection.
- This repository does not currently declare an open-source license.
