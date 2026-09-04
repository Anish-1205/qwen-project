"""Run the canonical stateful terminal interface for the local assistant.

This module intentionally performs model and index initialization at startup;
use ``infer.py`` for a stateless loop or ``webapp.py`` for the browser UI.
"""

import logging
import os
import sys
import warnings

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").disabled = True
warnings.filterwarnings("ignore", message="triton not found")

from huggingface_hub.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from documents import DocumentIndex
from harness import DEFAULT_SYSTEM_PROMPT, HarnessRunner, RunRequest
from logging_utils import assistant_label, capture_prints, dim_text, launch_log_tailer, prompt_text, setup_debug_logger, status_text, turn_status_text
from memory_core import OfflineMemoryManager
from models import create_backend, get_model_spec
from tool_selectors import configured_tool_selector

# Application budget for prompt compression, not the model's hard context limit.
MAX_CONTEXT_TOKENS = 1500
KEEP_RECENT_TURNS = 2

logger, log_path = setup_debug_logger()
if not launch_log_tailer(log_path, logger):
    print(status_text(f"[Status] Debug log: {log_path}"))

with capture_prints(logger):
    logger.info("Loading model...")
    model_selection = os.environ.get("CHATBOT_MODEL", "qwen").strip().lower()
    model_backend = create_backend(get_model_spec(model_selection))
    model_backend.load()
    logger.info("Model loaded.")

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

with capture_prints(logger):
    memory = OfflineMemoryManager()
    document_index = DocumentIndex(memory.embed_model, logger=logger)
    document_index.sync()
    tool_selector = configured_tool_selector(logger=logger.info)

orchestrator = HarnessRunner(
    model_backend,
    memory,
    system_prompt=SYSTEM_PROMPT,
    compression_enabled=True,
    max_context_tokens=MAX_CONTEXT_TOKENS,
    keep_recent_turns=KEEP_RECENT_TURNS,
    reply_generation_kwargs={
        "max_new_tokens": 300,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    router_generation_kwargs={
        "max_new_tokens": 120,
        "do_sample": False,
    },
    document_lookup=document_index.lookup_context,
    tool_selector=tool_selector,
    logger=logger.info,
)

turn_number = 0

while True:
    user_input = input(prompt_text("You: ")).strip()
    if user_input.lower() in ("exit", "quit"):
        model_backend.close()
        print(status_text("Ending chat."))
        break
    if not user_input:
        continue

    turn_number += 1
    with capture_prints(logger):
        result = orchestrator.run(RunRequest(
            user_input=user_input,
            messages=messages,
            turn_number=turn_number,
            maintain_history=True,
        ))
        messages, reply = result.messages, result.output

    print(f"{assistant_label('Assistant:')} {reply}")
    print(turn_status_text(f"[Turn {turn_number}] logged."))
    sys.stdout.write("\n")
    sys.stdout.flush()
