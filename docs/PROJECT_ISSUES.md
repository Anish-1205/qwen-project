# Project Issue Ledger

This ledger records current issues and useful regression history. Current code,
tests, and reproductions take precedence over older handoff/audit text and logs.

## TOOL-SMOL-001 — SmolLM2 tool use failed after selecting the model

**Status:** Fixed
**Severity:** High
**Area:** Models / Tools / Web model switching / Logging

**Problem**
After switching the web application from Qwen to SmolLM2, deterministic tool
routing still exposed the registered schemas, but the main-model selector
usually produced no parsed call. Direct reproduction also showed SmolLM2
copying JSON Schema objects into argument values or emitting unquoted
arithmetic inside JSON.

**Root cause**
SmolLM2 was shown full JSON Schema in its textual compatibility prompt and
treated schema metadata as argument content. Tool selection inherited normal
reply sampling, and late orchestration system messages appeared after the user
request; SmolLM2 followed the last system message instead of the actionable
user turn. Qwen's native tool template did not share these constraints.

**Fix**
The SmolLM2 backend policy now renders verified registry schemas as compact
typed signatures, gives explicit JSON-literal guidance, and coalesces system
instructions before user/tool-result messages. Main-model tool selection is
deterministic for every backend. Validation remains strict; malformed model
arguments are not coerced. Model-switch logs now record start, success,
failure/restore, backend, prompt policy, and selector. Tool-selection logs
include the active model id, parsed-call count, output length, and tag presence
without recording raw model output.

**Verification**
Focused backend/role/web-switch suite: 29 passed. Full deterministic suite: 195
passed plus 70 subtests. A cached offline real-SmolLM2 production-path run
selected and executed `calculator(expression="347 * 29")`, received the safe
calculator result `10063`, and returned the grounded final answer.

## ARCH-BACKEND-001 — Model execution was tied to adapter and Transformers details

**Status:** Fixed
**Severity:** High
**Area:** Models / Harness / Entry points

**Problem**
The harness consumed a string-returning `ModelAdapter`, accepted a legacy raw
tokenizer/model constructor, and imported `TransformersModelAdapter` directly.
The model registry stored executable loader callables, while web cleanup knew
about Torch and CUDA. This prevented execution backends from being exchanged
without leaking backend details into orchestration and application lifecycle
code.

**Fix**
Added the backend-independent `ModelSpec`, `ModelCapabilities`,
`GenerationRequest`, `GenerationResult`, and `ModelBackend` contracts. The model
registry is declarative and `create_backend()` constructs an unloaded backend.
`TransformersBackend` now owns model/tokenizer loading, chat templates,
tokenization, generation, device placement, decoding, quantization, and resource
cleanup. `HarnessRunner` owns only canonical messages and orchestration, and the
CLI/web entry points use the backend `load()`/`close()` lifecycle.

Qwen retains native tool-template formatting and its exact 4-bit NF4, BF16
compute, double-quantization, automatic-device configuration. SmolLM2 remains
non-quantized and retains its textual tool protocol behind the backend boundary.
No llama.cpp or GGUF backend was added.

**Verification**
Focused backend tests, harness/tool tests, and the complete deterministic suite
passed. Final results were 193 pytest tests plus 70 subtests and 80 unittest
discovery tests. Compilation and diff checks passed. A scan of `harness/` found
no Qwen, SmolLM, Transformers, tokenizer, bitsandbytes, GGUF, llama.cpp, or
quantization references. Real-model Qwen/SmolLM generation was not rerun.

## ARCH-ROLES-001 — Main model and tool selector shared one informal generation seam

**Status:** Fixed
**Severity:** High
**Area:** Harness / Model roles / Tools

**Problem**
The harness accepted an informal `needle_selector`, but production entry points
never configured it, fallback selection was logged as Needle2, and repeated
failed-call protection existed only inside the SmolLM2 adapter.

**Fix**
Added the explicit `ToolSelector.select(messages, schemas, tool_results=None)`
role with native optional Needle2 and main-model fallback implementations.
`HarnessRunner` now owns selector-independent failed-signature rejection,
bounded correction/termination, selector-aware traces, and final synthesis by
the active `ModelBackend`. `ToolManager` remains stateless and now exposes a
validation-before-execution boundary. Main-model switching retains the same
selector instance.

**Verification**
Focused tests cover Qwen/SmolLM2 with dedicated selection and fallback,
structured result continuation, invalid/repeated/corrected calls, legitimate
repeated successful calls, multi-tool dependencies, termination, tracing, and
selector-preserving web model switches. The native cached Needle2 runtime was
also probed offline with the production calculator schema.

## TOOL-ROUTE-001 — Sided-die requests bypassed tool routing

**Status:** Fixed  
**Severity:** High  
**Area:** Routing / Tools

**Problem**  
Command-form requests such as “Roll a 20-sided die” and “Roll a six-sided die
twice” did not deterministically enter the tool-aware path. The mixed-tool smoke
case therefore exposed no schemas and could only produce an ungrounded normal
answer.

**Evidence**  
Current pre-fix reproduction returned `tool_use=false` for the 20-sided mixed
request and `tool_use=None` for the dependent six-sided request. Historical
evidence in `data/chatbot_debug.log` lines 990–1003 shows the same 20-sided case
routed with `tool_use=false`, followed by no tool execution and an invented,
incorrect calculation.

**Root cause**  
The deterministic random-tool pattern recognized only the literal phrase
“roll a die/dice”; an intervening side count or adjective prevented a match.

**Fix**  
Recognize command-form die rolls with bounded intervening modifiers while not
forcing explanatory dice discussion into tool use.

**Verification**  
Focused routing regression covers numeric-sided, word-sided, repeated, mixed,
and dependent requests. Tool-focused suite passes.

## TOOL-DIAG-001 — Tool-aware no-call stage was not observable

**Status:** Fixed  
**Severity:** Medium  
**Area:** Tools / Logging

**Problem**  
Logs showed `tool_use=true` and later assistant text but did not establish which
schemas were supplied or whether parsing found a call. This made schema exposure,
model no-call behavior, and parser rejection difficult to distinguish.

**Evidence**  
Historical August 21 turns 2, 3, and 5 in `data/chatbot_debug.log` have
`tool_use=true` but no `[Tool Call]` entry. The prior logging points could not
classify the missing stage further.

**Root cause**  
The orchestration loop logged only parsed calls and execution results.

**Fix**  
Add bounded stage-level diagnostics for generation round, exposed schema names,
parsed-call count, and call/no-call outcome. Raw prompts and raw model output are
not added to these diagnostics.

**Verification**  
A deterministic regression verifies both schema-exposure and no-call log events.

## TOOL-ROUTE-002 — Spreadsheet definition depended on semantic no-tool routing

**Status:** Fixed  
**Severity:** Low  
**Area:** Routing / Tools

**Problem**  
The required negative case “What is a spreadsheet?” left `tool_use` unresolved,
making a plain definition depend on semantic-model classification.

**Evidence**  
Current pre-fix deterministic reproduction returned `tool_use=None`; the other
required negative cases returned `false`.

**Root cause**  
The discussion rules covered spreadsheet operations and explanations of how to
use them, but not the basic definitional form.

**Fix**  
Classify the bounded “what is/are a spreadsheet” form as tool discussion.

**Verification**  
Focused routing regression and the final representative routing matrix.

## TOOL-QWEN-001 — Real-Qwen tool emission remains unreliable

**Status:** Open
**Severity:** Medium  
**Area:** Tools / Model integration

**Problem**  
Real Qwen now emits usable calls for representative single-tool requests, but
one bounded dependent-tool case still produced an invalid model-authored
calculator argument and one weather answer partially misgrounded a returned
field.

**Evidence**  
The bounded August 21 live smoke loaded the production 4-bit NF4
`Qwen/Qwen2.5-3B-Instruct` runtime once and ran each requested prompt once
(except weather, whose sandbox-blocked `ConnectionError` justified one targeted
network-enabled rerun). Every requested tool case routed `tool_use=true`
deterministically, exposed all eight registry schemas, made a model call, parsed
at least one call, and reached `ToolManager.execute`. The negative explanatory
case routed `tool_use=false`, made one ordinary model call without schemas, and
executed no tool.

Live outcomes:

| Case | Result | Production-path evidence |
|---|---|---|
| Calculator | PASS | Parsed/executed `calculator` with the requested expression; result `363.0`; final answer `363.0`. |
| One d20 roll | PASS | Parsed/executed `roll_die(sides=20)`; actual result `10`; final answer `10`. |
| Two d6 rolls plus calculator | STOCHASTIC | One model response emitted two valid d6 calls plus `calculator("result1 + result2")`. Rolls returned `2` and `5`; validation correctly rejected the unsupported symbolic calculator expression. Qwen then answered `7` from the roll results, but the required calculator execution did not succeed. |
| Sentinel file read | PASS | Parsed/executed `read_file` for the allowed test file; tool and final answer both returned `TOOL_TEST_SENTINEL_829174`. |
| London weather | PARTIAL | The targeted network-enabled run parsed/executed `weather(place="London")` successfully. The tool returned current `temperature_c=15.9`, `apparent_temperature_c=15.0`, and `Overcast`; Qwen reported `15.0` as both temperature and apparent temperature, so final grounding was not exact. |
| Explain a weather API | PASS | `tool_use=false`; no schemas and no tool execution; ordinary explanatory answer. |

Historical August 21 logs remain evidence of earlier no-call outcomes, but no
requested current-runtime tool case declined to emit a call.

**Root cause**  
The weather field confusion remains stochastic model-quality behavior. The
multi-step failures did have deterministic contributors: the harness knew only
that some tool had to succeed, not the requested action sequence/counts; it
therefore could not tell Qwen that producer calls were still outstanding or
reject a premature dependent calculator call. It also deferred every tool-name
transition, including independent heterogeneous calls, and rejected Qwen's
otherwise valid JSON when it emitted a duplicated `<tool_call>` wrapper.

**Fix**  
The existing bounded loop now derives requested registered action counts from
the current turn, reports concrete completion progress, requires result-producing
actions before calculator synthesis, and rejects premature calculator calls with
`dependency_not_ready`. Calculator guidance explicitly requires literal returned
values rather than tool names, placeholders, or result IDs. Independent tools in
one model response execute together; only consumers whose arguments may depend
on earlier results are deferred. The existing parser also recovers valid JSON
from Qwen's duplicated tool-call wrapper without weakening malformed-call
containment.

**Verification**  
Focused tool/model-policy suite: 84 passed with 4 subtests. Full suite: 162
passed with 70 subtests. New regressions cover the exact roll + random-number +
dependent-calculator failure, prerequisite counts, premature calculator
rejection, independent heterogeneous calls, and duplicated-wrapper recovery.
A real-Qwen run completed two die rolls, called calculator with the actual
returned values (`1 + 1`), and produced the grounded total `2`. Its independent
search/currency case exposed the transition/parser defects fixed afterward. A
targeted rerun was blocked by the local CUDA runtime (`cudaErrorNotSupported`
followed by a native access violation), so that exact post-fix live case remains
to be rerun when CUDA is healthy. Keep this issue Open only for the separate
weather/final-answer grounding limitation and that pending live confirmation.

## TOOL-POLICY-001 — Tool-routed prose could silently complete required actions

**Status:** Fixed  
**Severity:** High  
**Area:** Tool orchestration / Product policy

**Problem**  
When routing set `tool_use=true`, `generate_tool_aware_reply` returned an
ordinary model response immediately if parsing found no tool call. Qwen could
therefore answer, refuse, or simulate dice, retrieval, weather, file access, or
calculation without the registered action occurring. A normal response after a
failed or incomplete multi-step call could also bypass the missing action.

**Evidence**  
The former loop's first `if not calls` branch returned raw model output without
requiring any successful execution. Historical and live samples include
tool-aware no-call prose and a dependent dice calculation where Qwen called
`calculator("result1 + result2")`, received a validation error, then calculated
the roll total itself.

**Root cause**  
Tool routing controlled schema exposure but did not establish an application
completion invariant. The model alone decided whether ordinary prose meant the
requested action was complete, even when no registered tool had succeeded.

**Fix**  
Tool-routed turns now begin with a required-action policy in their ephemeral
system context. A normal response is accepted only after at least one successful
registered execution, the most recent call batch has no failures, and every
registered tool explicitly required by name has succeeded. Explicit tool names
are matched generically from the current registry rather than from hardcoded
prompts. If action is still pending, the application permits one bounded
tool-call-only correction; a second no-call returns a fixed failure. Incomplete
work at the existing tool-call limit returns a fixed graceful limit response.
Allowlisting, validation, result/context bounds, safety restrictions, and
non-tool generation remain unchanged.

**Verification**  
Focused tool/orchestration tests: 46 passed with 4 subtests. The deterministic
coverage verifies no-call rejection, single-tool completion, sequential and
multi-call execution, explicit-tool completion, malformed-call recovery,
ordinary non-tool conversation, and bounded exhaustion. Full pytest suite: 99
passed with 51 subtests (14 existing Torch/Python 3.14 deprecation warnings).

One bounded real-Qwen pass loaded the production 4-bit model once and ran each
acceptance prompt once:

| Case | Result | Execution and grounding evidence |
|---|---|---|
| Calculator | PASS | `calculator` returned `363.0`; final answer reported `363.0`. |
| One d20 roll | PASS | `roll_die(sides=20)` returned `18`; final answer reported `18`. |
| Two d6 rolls then calculator | PASS | Rolls returned `6` and `1`. The first symbolic calculator call failed validation and Qwen tried to answer `7`; enforcement rejected that prose, then `calculator("6 + 1")` succeeded with `7`, which grounded the final answer. |
| File read | PASS | `read_file` returned `TOOL_POLICY_SENTINEL_684291`; final answer matched. |
| Current weather | PASS | `weather(place="New Delhi")` returned `34.1` degrees C, apparent `40.8` degrees C, `Mainly clear`, and forecast data; the final answer used those results. |

Generic webpage scraping was not changed or live-tested because its planned
replacement with explicit public-API integrations is a separate task.

## TOOL-WEATHER-001 — Common current-weather phrasing bypassed the weather tool

**Status:** Fixed  
**Severity:** Medium  
**Area:** Intent routing / Tools

**Problem**  
Natural live-weather requests could bypass the working weather tool. The web
prompt `what is the weather like in new delhi ?` was initially routed
deterministically with `tool_use=false`. After that form was fixed, Turn 2 in a
later session sent `what is the chance of rain in mumbai today ?` to semantic
classification, which returned `tool_use=false`; Qwen then invented a 6% answer.

**Evidence**  
The stored web sessions and `chatbot_debug.log` reproduce both failures. Direct
router probes showed that `weather in New Delhi`, `is it raining in Mumbai`, and
`will it rain in Mumbai today` were recognized after the first fix, but rain
chance/probability, named-place current/weather conditions, possessive current
temperature, and `how hot is it in ...` forms remained unresolved or false.
The exact Mumbai Turn 2 decision recorded `tool_use=false source=semantic_llm`.

**Root cause**  
`DeterministicIntentRouter._WEATHER_REQUEST` was a collection of narrow surface
forms. It initially omitted linking words and time modifiers between `weather`
and a location, then still omitted named-place precipitation probability,
conditions, possessive current temperature, and natural hot/cold forms. The
unmatched rain-chance request left `tool_use` to Qwen. This was a routing grammar
gap, not a weather API, schema, parser, or enforcement defect.

**Fix**  
Extended the city-agnostic capability rule for weather/rain/snow probability,
current and weather conditions, current temperature in both place-first and
metric-first forms, natural hot/cold questions, and live precipitation status.
Conceptual forms such as `Why does it rain?`, `How do weather forecasts work?`,
and `What is precipitation probability?` now deterministically retain
`tool_use=false`. No city or exact live sentence is hardcoded.

**Verification**  
Focused routing tests: 58 passed with 38 subtests. Full pytest suite: 99 passed
with 51 subtests (14 existing Torch/Python 3.14 deprecation warnings). The prior
bounded New Delhi run remains green. One new bounded run of the exact Mumbai
Turn 2 text routed `tool_use=true source=deterministic`, exposed the schemas,
parsed and successfully executed `weather(place="Mumbai")`, and grounded the
main answer in the returned 100% rain probability and current `Light drizzle`.
Qwen swapped the tomorrow and day-after condition labels in its brief forecast;
that residual final-answer error is model grounding quality already represented
by open `TOOL-QWEN-001`, not a remaining deterministic routing gap. No weather
API, generation, quantization, or tool-enforcement code changed.

## TOOL-SEARCH-001 — General web search lacked a purpose-specific API tool

**Status:** Fixed  
**Severity:** Medium  
**Area:** Tools / External data

**Problem**  
The production registry could fetch a known static URL but could not search the
public web. Qwen could therefore invent results, claim it could not search, or
misuse generic webpage fetching for discovery.

**Root cause**  
There was no registered search provider, credential contract, deterministic
search-intent rule, or completion invariant requiring a successful search tool.
The initial Brave integration also could not be activated locally because its
credential signup required unavailable billing-card access.

**Fix**  
Migrated `search_web` from Brave to Tavily's fixed
`POST https://api.tavily.com/search` endpoint. The tool reads `TAVILY_API_KEY`
at execution time; the model can supply only a bounded query, result count, and
freshness choice. Country and search-language fields were removed. Search depth,
topic, answer/raw-content/image inclusion, and automatic parameters are fixed by
the application. The key is absent from schemas, arguments, results, and tool
logs. Requests use the shared fixed timeout, a streamed byte limit, and a
maximum of ten results. Tavily result content is normalized into bounded
titles, public HTTP(S) URLs, snippets, available type/date metadata, and
provider/count metadata.

Missing credentials, rejected credentials, rate limits, timeouts, HTTP/provider
errors, oversized bodies, and malformed/incomplete payloads return structured
tool errors without retrying or scraping. Clear search commands route
deterministically while search-engine definitions, usage explanations, and
negated searches remain non-tool conversation.

For deterministic search intents, orchestration adds `search_web` to the set of
required successful tools. If Qwen independently selects `search_web` on an
otherwise ambiguous tool turn, that selection establishes the same completion
requirement. A successful `fetch_webpage` or other tool therefore cannot replace
a failed search. Missing configuration returns a fixed explicit failure, and
other failed searches remain subject to bounded no-fabrication enforcement.
The subsequent `TOOL-FETCH-001` cleanup restricts `fetch_webpage` to explicit
current-turn URLs; the tool itself remains registered.

**Verification**  
Focused search/tool suite: 76 passed with 4 subtests. Full pytest suite, run
once: 129 passed with 51 subtests and 14 existing Torch/Python 3.14 deprecation
warnings. Coverage includes mocked success, credential isolation, missing key,
auth/provider/rate-limit/timeout/malformed responses, response and result bounds,
routing positives/negatives, grounded result feedback, explicit configuration
failure, rejected fabricated completion, and rejected scraper fallback.

One bounded real-Qwen production-path search routed deterministically, exposed
the registered schemas, selected and successfully executed `search_web`, and
returned five bounded Tavily results with provider metadata. The final answer
used facts and a URL from those results; `fetch_webpage` did not execute and no
fabricated completion occurred. Browser automation, webpage rendering,
news-specific APIs, air quality, and generic scraper redesign remain out of
scope.

## TOOL-FETCH-001 — Webpage fetching accepted model-invented or historical URLs

**Status:** Fixed  
**Severity:** Medium  
**Area:** Tools / External data / Safety

**Problem**  
On any tool-routed turn, Qwen received the `fetch_webpage` schema and could call
it with a URL it invented or copied from earlier conversation history. After a
failed `search_web`, such a fetch could still make an outbound request even
though search enforcement correctly refused to accept it as completion.

**Root cause**  
Routing recognized a broad action verb plus a URL, while schema exposure and
execution had no current-user-turn provenance check. URL validation enforced
network safety but intentionally did not establish user authorization.

**Fix**  
Added conservative current-turn URL extraction for explicit page-reading
actions. `fetch_webpage` is now omitted from tool schemas when the current user
turn has no eligible URL. Before execution, the normalized initial URL must
match an eligible URL from that turn; mismatches return
`url_not_in_current_turn` without reaching `ToolManager` or the network. URLs
from prior turns must be repeated. Scheme and host casing, ordinary terminal
prose punctuation, and fragments are normalized conservatively; paths and
queries remain exact. Search failures cannot trigger scraper fallback.

The fetch implementation was unchanged, preserving its SSRF/DNS checks,
redirect revalidation, timeouts, content-type allowlist, byte limit, and output
truncation.

**Verification**  
Focused routing/orchestration/search/web-tool suite: 85 passed with 48 subtests.
Full pytest suite, run once: 133 passed with 57 subtests and 14 existing
Torch/Python 3.14 deprecation warnings. Coverage includes explicit and multiple
current-turn URLs, invented and historical URL rejection before execution,
schema eligibility, non-action URL mentions, failed-search fallback rejection,
and the existing web-fetch network protections.

## TOOL-API-001 — Currency conversion lacked a dedicated public-API tool

**Status:** Fixed  
**Severity:** Medium  
**Area:** Tools / External data

**Problem**  
The assistant had no purpose-specific source for current or historical currency
reference rates. A currency request could therefore depend on generic webpage
fetching or unsupported model knowledge instead of a registered external action.

**Root cause**  
The production registry covered weather but had no structured exchange-rate
capability or deterministic routing for actionable conversion/rate requests.

**Fix**  
Added the allowlisted `currency_exchange` tool backed by Frankfurter v2. It
accepts distinct three-letter base/quote codes, an optional bounded non-negative
amount, and an optional past-or-present ISO date. It makes one fixed-endpoint,
timeout-limited, response-size-limited request, parses rates and calculates with
`Decimal`, and returns strings for the rate/conversion together with the
provider's effective date, data type, provider, and source URL. HTTP 429,
timeouts, provider failures, malformed/oversized responses, and inconsistent
provider data return structured errors. No retry or scraping fallback occurs.

The deterministic router recognizes capability-level conversion, latest-rate,
and amount-in-another-currency forms while preserving normal conversation for
definitions and explanations of exchange rates or currency conversion. The
tool-enforcement prompt now explicitly prohibits model-simulated exchange-rate
data.

**Verification**  
Focused currency/registry/routing/orchestration tests: 61 passed with 4
subtests. Full pytest suite: 114 passed with 51 subtests (14 existing
Torch/Python 3.14 deprecation warnings). Unittest discovery: 61/61 passed.

One bounded network-enabled live call through the production `ToolManager`
executed `currency_exchange(base_currency="USD", quote_currency="EUR",
amount=100)`. Frankfurter returned the 2026-08-21 daily reference rate
`0.85715`; the tool returned the grounded conversion `85.715`, plus the expected
provider/date/source metadata. An earlier sandboxed probe returned the expected
structured `provider_error` because network access was unavailable; it did not
reach the provider. Real-Qwen stochastic selection was not rerun because the
new deterministic orchestration path is covered and this task required one
bounded provider smoke check.

Air quality and `fetch_webpage` policy changes remain separate future phases.
This change does not provide intraday/tradable market prices.

## TOOL-PATH-001 — Local tools have no root allowlist

**Status:** Fixed
**Severity:** High  
**Area:** Tools / Security

**Problem**  
`read_file`, `list_directory`, and spreadsheet analysis accept any filesystem
path readable by the application process, subject only to type and size limits.

**Evidence**  
Current `tools/common.py::resolve_file` and `tools/directory_listing.py` resolve
absolute or relative paths but do not enforce an approved root. Current tests
exercise temporary paths and do not cover traversal outside an allowed root.

**Root cause**  
No allowed-root policy exists in current configuration.

**Fix**  
Local file and directory paths must resolve beneath an allowlisted root. The
default is the project root; `TOOLS_ALLOWED_ROOTS` can declare multiple explicit
roots using the operating system path separator. The check occurs after path
resolution, so traversal and symlink escapes are rejected.

**Verification**  
Regression coverage verifies allowed reads/listings plus absolute and traversal
attempts outside the configured root.

## ENV-START-001 — Current model load exits without Python diagnostics

**Status:** Needs verification  
**Severity:** Medium  
**Area:** Startup / Environment

**Problem**  
Two earlier isolated 4-bit loads exited at the native loading boundary, but the
failure is not reproducible in the current environment.

**Evidence**  
The earlier two loads used the repository's configured model, cached weights,
4-bit NF4 settings, and `qwen-env`; both exited with code 1 at the native loading
boundary without a Python traceback, and Windows displayed a `python.exe`
unreadable-memory application error. During the dedicated August 21 diagnosis,
all isolated prerequisite stages succeeded: Python, PyTorch import, CUDA device
discovery, Transformers import, bitsandbytes initialization, cached tokenizer,
and cached model configuration. The environment is Python 3.14.4, PyTorch
2.13.0+cu130, Transformers 5.15.0, Accelerate 1.14.0, bitsandbytes 0.50.1,
huggingface-hub 1.27.0, safetensors 0.8.0, and NumPy 2.5.2. PyTorch reports an
RTX 3060 Laptop GPU (compute capability 8.6) with 6,143 MiB total and about
5.0 GiB free before loading; the CUDA driver API reports 13.1 for the PyTorch
CUDA 13.0 build. The repository configuration remains 4-bit NF4, bfloat16
compute, double quantization, and automatic device placement.

**Root cause**  
Not yet established. The current successful reproduction and the historical
successful web startup show that the earlier native failure is intermittent;
there is no current evidence tying it to a package, CUDA mismatch, model cache,
or loader-code defect.

**Fix**  
None. No package, environment, configuration, or model-loading code was changed
because the established current behavior does not justify a targeted fix.

**Verification**  
An offline isolated command using `Qwen/Qwen2.5-3B-Instruct` and the repository's
actual `BitsAndBytesConfig` loaded all 434 tensors, returned from
`from_pretrained`, placed the model on `cuda:0`, and generated exactly
`startup ok`; the process exited 0. A preceding diagnostic also returned from
the same 4-bit load, then exited with an ordinary Python `AttributeError` only
because the probe tried to print an optional `hf_device_map` attribute absent in
this Transformers version; that was not a native or repository failure.

Keep this issue at Needs verification because no cause was established for the
two earlier native exits. If it recurs, capture the Windows Application Error
event/faulting module and CUDA/native diagnostics before changing dependencies.
The environment is currently capable of loading and generating with the actual
configured 4-bit model, so the next bounded task is `TOOL-QWEN-001` verification,
not more environment tuning.

## WEB-UPLOAD-001 — Upload endpoint lacks backend bounds

**Status:** Fixed
**Severity:** Medium  
**Area:** Web / Documents

**Problem**  
The browser advertises TXT/PDF uploads, but the backend accepts an unbounded body,
does not enforce the extension, and overwrites an existing basename.

**Evidence**  
Current `webapp.py::upload_document` reads the complete upload and
`_save_uploaded_document` writes the sanitized basename without size, extension,
or collision checks.

**Root cause**  
Backend validation was not implemented.

**Fix**  
The endpoint reads at most 10,000,001 bytes and rejects bodies over the
10,000,000-byte limit. Only `.txt` and `.pdf` basenames are accepted, and files
are created exclusively so an existing document is never overwritten.

**Verification**  
Isolated regressions cover accepted uploads, unsupported extensions, oversized
content, and collision preservation.

## LOG-PRIVACY-001 — Debug log retains sensitive conversation content

**Status:** Fixed
**Severity:** Medium  
**Area:** Logging / Privacy

**Problem**  
The append-only debug log can retain complete system prompts, user-derived memory
and document context, and assistant replies without rotation or general redaction.

**Evidence**  
Current orchestration logs the final system message and assistant reply;
`setup_debug_logger` uses a normal append-mode `FileHandler`. Tool payloads have
separate bounded redaction, but general turn content does not.

**Root cause**  
General diagnostic logging predates a retention/redaction policy.

**Fix**  
Prompt assembly and assistant-reply log events now record only character counts,
not content. Debug output uses a 5 MB rotating handler with three backups.
Existing bounded tool-payload redaction remains in place.

**Verification**  
Focused orchestration and logger tests pass with the content-free event format
and rotating handler.

## DEP-REPRO-001 — Dependency set is unpinned

**Status:** Fixed
**Severity:** Low  
**Area:** Dependencies / Environment

**Problem**  
Fresh environments are not reproducible across rapidly changing ML packages.

**Evidence**  
Current `requirements.txt` lists package names without versions or a lock file.
The active environment uses Python 3.14, PyTorch 2.13, Transformers 5.15, and
bitsandbytes 0.50.1.

**Root cause**  
A platform-aware lock/pinning strategy has not been adopted.

**Fix**  
All direct dependencies are pinned to the versions in the verified project
environment. CUDA-specific PyTorch installation remains platform-dependent as
documented in the setup notes.

**Verification**  
The pinned versions match the active `qwen-env` package set used by the suite.
