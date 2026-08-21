"""Run the stateless terminal interface for one-turn-at-a-time inference."""

import logging
import sys
import warnings

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").disabled = True
warnings.filterwarnings("ignore", message="triton not found")

from huggingface_hub.utils import logging as hf_logging

hf_logging.set_verbosity_error()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from documents import DocumentIndex
from logging_utils import assistant_label, capture_prints, dim_text, launch_log_tailer, prompt_text, setup_debug_logger, status_text, turn_status_text
from memory_core import OfflineMemoryManager as BaseOfflineMemoryManager
from orchestrator import ConversationOrchestrator, DEFAULT_SYSTEM_PROMPT, strip_speaker_tags

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


class OfflineMemoryManager(BaseOfflineMemoryManager):
    pass


def postprocess_reply(text: str) -> str:
    return strip_speaker_tags(text)


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger, log_path = setup_debug_logger()
    if not launch_log_tailer(log_path, logger):
        print(status_text(f"[Status] Debug log: {log_path}"))

    with capture_prints(logger):
        logger.info("Loading Qwen2.5-3B-Instruct model in 4-bit NF4...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()
        logger.info("Model loading complete.")

        memory = OfflineMemoryManager()
        document_index = DocumentIndex(memory.embed_model, logger=logger)
        document_index.sync()

    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    orchestrator = ConversationOrchestrator(
        tokenizer,
        model,
        memory,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        compression_enabled=False,
        reply_generation_kwargs={
            "max_new_tokens": 450,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        router_generation_kwargs={
            "max_new_tokens": 120,
            "do_sample": False,
        },
        document_lookup=document_index.lookup_context,
        logger=logger.info,
    )

    turn_number = 0

    while True:
        try:
            user_input = input(prompt_text("You: ")).strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print(status_text("Ending session."))
                break

            turn_number += 1
            with capture_prints(logger):
                messages, reply, _, _ = orchestrator.process_turn(
                    user_input,
                    messages,
                    turn_number=turn_number,
                    maintain_history=False,
                    reply_postprocess=postprocess_reply,
                )

            print(f"{assistant_label('Assistant:')} {reply}")
            print(turn_status_text(f"[Turn {turn_number}] done."))
            sys.stdout.write("\n")
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n" + status_text("Ending session."))
            break
        except Exception as e:
            print(f"\n{status_text(f'Error: {e}')}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
