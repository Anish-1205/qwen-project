# AGENTS.md

## Mission

Work on this repository as a production-focused senior software engineer.

Prioritize, in order:

1. Correctness and preservation of existing behavior.
2. Root-cause fixes over symptom patches.
3. Small, reviewable, maintainable changes.
4. Regression protection through tests.
5. Privacy, data integrity, and safe local execution.
6. Clear reporting of what changed, what was verified, and what remains uncertain.

Do not optimize for speed at the expense of correctness.

---

## Source of truth

The historical repository audit is not the current source of truth.

Before making a meaningful change:

1. Read `docs/PROJECT_HANDOFF.md` if it exists.
2. Read `docs/PROJECT_ISSUES.md` if it exists.
3. Inspect the current repository tree and relevant implementation/tests.
4. Check `git status` and do not overwrite unrelated user changes.
5. Treat old audit documents, verification artifacts, experiments, logs, and compatibility modules as historical evidence unless current code confirms otherwise.

Current architecture should be inferred from the repository, not from stale filenames.

In particular, expect current execution code to live under `harness/`, model adapters under `models/`, and treat an old `orchestrator.py` path as compatibility-only unless current code proves otherwise.

---

## Before coding

For every non-trivial bug fix or feature:

- Restate the intended behavior from the issue/request in implementation terms.
- Trace the relevant call path before editing.
- Find the existing tests that cover the behavior.
- Identify persistence, model, routing, memory, RAG, tool, session, logging, and web-boundary effects where relevant.
- Prefer extending existing abstractions over creating parallel implementations.
- Do not invent APIs, schema fields, config keys, or dependencies without verifying them in the repository.
- Do not perform broad refactors unless they are required for the requested change.

If the problem is ambiguous, inspect more code before changing anything.

---

## Debugging standard

When fixing a bug:

1. Reproduce or characterize the failure first when practical.
2. Identify the root cause.
3. Add or update a regression test that fails for the old behavior when practical.
4. Implement the smallest coherent fix.
5. Run the focused test first.
6. Run the broader relevant suite afterward.
7. Review the final diff for accidental behavior changes.

Never:

- delete or weaken tests just to make the suite pass;
- suppress exceptions without understanding them;
- broaden fallbacks in ways that hide malformed model output;
- replace validation with permissive parsing merely to satisfy a failing case.

---

## Feature implementation standard

For new features:

- Match current architecture and naming.
- Preserve backward compatibility unless the task explicitly changes a contract.
- Validate inputs at trust boundaries.
- Handle failure paths explicitly.
- Add tests for normal behavior, edge cases, and invalid inputs.
- Update user-facing or developer documentation when behavior/configuration changes.
- Avoid introducing unnecessary dependencies.

For model-facing behavior, distinguish deterministic application guarantees from probabilistic model behavior.

---

## Repository-specific safety constraints

### Persistent data

Treat runtime databases and user data as sensitive.

Unless the user explicitly asks for mutation of live data:

- do not modify production/runtime SQLite databases;
- use temporary databases, fixtures, or read-only inspection for tests and debugging;
- do not delete or rewrite persisted sessions, memory, documents, or user data as part of validation;
- do not treat WAL/SHM/runtime artifacts as source files.

For schema or migration changes, create an explicit migration/upgrade path and test it against disposable copies.

### Logging and privacy

Assume logs may contain sensitive content such as:

- system prompts;
- user messages;
- memory facts;
- retrieved document chunks;
- tool arguments/results;
- assistant replies.

Do not add new logging of sensitive full-context data unless necessary and explicitly justified.

Prefer:

- metadata over raw content;
- configurable redaction;
- bounded/rotating logs;
- no secrets or credentials in logs.

### Web exposure

The application is intended to remain local-only unless authentication and network-facing protections are deliberately added.

Do not broaden bind addresses, disable localhost restrictions, or expose the service to a network as an incidental change.

When touching uploads or web inputs, enforce backend validation rather than relying only on UI/client validation.

### Tools

Tool execution must remain allowlisted and validated.

Do not:

- execute unknown tool names;
- bypass schema validation;
- replace restricted arithmetic evaluation with `eval()` or equivalent arbitrary execution;
- silently coerce malformed arguments when a validation error is the safer contract.

Keep tool results structured and bounded.

### Memory

Preserve the principle that durable memory is grounded in user evidence.

Do not weaken:

- entity checks;
- relation/value validation;
- duplicate protection;
- conflict/invalidation semantics;
- grounding requirements.

Avoid persistence side effects from operations that are intended to be read-only.

### RAG / document retrieval

Preserve explicit filename scoping and avoid injecting unrelated semantic matches when a specifically named file is missing.

When changing retrieval thresholds, chunking, ranking, or budgets:

- add targeted regression tests;
- verify both relevant retrieval and decoy rejection;
- avoid tuning from a single example.

---

## Known risk areas to treat carefully

Check current issue/handoff docs before acting, but historically important risk areas include:

- reproducible dependency metadata;
- session reset and persistence semantics;
- sensitive log redaction/rotation;
- backend upload extension/size/overwrite validation;
- prompt/context budgeting;
- real-model tool-call regression coverage;
- heuristic routing failures;
- retrieval scaling and duplicate embedding work;
- localhost-only security assumptions.

Do not assume any historical risk is still present. Verify current code first.

---

## Model and orchestration changes

Changes involving model loading, adapters, generation parameters, routing, prompt construction, compression, or tool loops require extra care.

Before changing them:

- inspect all active entry points;
- identify shared vs entry-point-specific configuration;
- check whether behavior is deterministic or sampling-based;
- avoid duplicating configuration across modules;
- verify token/context budgets end to end.

Do not claim a model behavior regression is fixed solely from mocked tests if the bug depends on real generation behavior.

For real-model validation:

- keep tests non-destructive;
- make them opt-in when they are expensive or hardware-dependent;
- record exact model/configuration used;
- separate deterministic unit-test guarantees from smoke-test evidence.

---

## Testing and validation

Use the repository's current documented environment and commands.

Do not assume historical test counts or commands are still current.

Validation order:

1. Run the narrowest relevant test(s).
2. Run the relevant package/module suite.
3. Run the full discoverable unit suite when practical.
4. Run lint/type checks/build checks if configured.
5. Run real-model or web smoke tests only when relevant, safe, and practical.
6. Review `git diff` and `git status`.

If no dependency manifest or canonical test command exists, inspect the current repo and document the gap rather than guessing.

Never mutate production databases merely to validate a change.

---

## Dependency changes

Before adding a dependency:

- verify an existing dependency cannot solve the problem;
- prefer mature, maintained packages;
- pin or constrain versions according to the repository's chosen dependency strategy;
- update the canonical dependency metadata/lockfile;
- document platform-specific requirements where relevant.

Do not create multiple competing dependency-management systems.

---

## Refactoring rules

Refactor only when it reduces concrete risk or is required by the requested feature/fix.

When refactoring:

- preserve externally observable behavior;
- separate compatibility shims from canonical implementations;
- avoid moving unrelated files;
- update imports and tests atomically;
- keep commits/diffs conceptually focused.

Prefer removing obsolete coupling over adding new cross-package dependencies.

---

## Completion criteria

Do not report a task as complete until you have:

- inspected the final diff;
- run the relevant tests/checks;
- confirmed no unrelated files were modified intentionally;
- checked for obvious persistence/privacy/security regressions;
- updated tests and documentation where needed.

In the final response, report:

1. What changed.
2. Why it changed.
3. Files modified.
4. Tests/checks run and their results.
5. Any validation not run.
6. Remaining risks or follow-up items.

If something could not be verified, say so explicitly.

---

## Change discipline

Keep changes minimal and reversible.

Do not:

- rewrite large subsystems without need;
- modify generated/runtime artifacts;
- commit secrets, databases, logs, model weights, virtual environments, or caches;
- make destructive data changes without explicit approval;
- claim current behavior from historical artifacts without verifying the current revision.

When uncertain, inspect first and change second.

## Model harness architecture

The long-term architecture is backend-independent.

Code under `harness/` must not depend directly on:

- specific model families such as Qwen, SmolLM, Llama, Mistral, etc.;
- Hugging Face Transformers implementation details;
- bitsandbytes;
- GGUF/llama.cpp;
- model-specific chat syntax;
- model-specific tool-call syntax.

Those concerns belong behind model backend and adapter interfaces.

A new compatible model should normally require configuration,
capability detection, or a model adapter—not changes to orchestration.

A new execution technology (Transformers, llama.cpp, etc.) should be
implemented as a backend without changing higher-level harness logic.

Model capabilities must be explicit rather than inferred throughout
the application from model-name conditionals.

Quantization is a backend concern.

Do not claim universal model compatibility. The supported contract is:
compatible local causal language models supported by an installed
backend.