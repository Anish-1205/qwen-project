# Contributing

This is a local-first assistant whose most important invariants are privacy,
bounded execution, and separation between memory, documents, and chat history.

## Development setup

Create a virtual environment, install `requirements.txt`, and run the unit tests
before and after a change:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

Normal tests use fakes and temporary databases. Do not make the default test
suite download or initialize Qwen.

## Change guidelines

- Keep `HarnessRunner` shared by all three entry points and model-independent.
- Keep model loading, tokenization, device placement, and generation inside a
  `ModelBackend`; do not add backend- or model-specific behavior to `harness/`.
- Keep tool-selection models behind `ToolSelector`; turn-level retry, duplicate,
  budget, and termination policy belongs in `HarnessRunner`.
- Treat model-produced tool names and arguments as untrusted input.
- Add executable tools through the registry and include validation and tests.
- Keep production tools stateless; they must not write memory or session data.
- Preserve separate SQLite stores for memory, documents, and web sessions.
- Never commit runtime databases, logs, model weights, caches, or local documents
  containing private information.
- Prefer comments that explain intent, constraints, or security reasoning. Avoid
  comments that simply repeat the code.
- Update the README and relevant files in `docs/` when behavior or configuration
  changes.

## Verification

Run focused tests while developing, followed by the complete suite:

```powershell
python -m pytest tests/test_harness.py tests/test_tools.py
python -m pytest
```

GPU-backed smoke tests are opt-in and should be reported separately. The current
architecture, risk register, and deeper verification notes live in
`docs/PROJECT_AUDIT_FULL.md`.
