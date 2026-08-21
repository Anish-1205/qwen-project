import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-3B"

# --- Tiny synthetic dataset, generated in-code — no separate file to manage ---
SAMPLE_TEXTS = [
    "Q: What is the capital of France?\nA: The capital of France is Paris.",
    "Q: What is 2+2?\nA: 2+2 equals 4.",
    "Q: Who wrote Romeo and Juliet?\nA: William Shakespeare wrote Romeo and Juliet.",
    "Q: What color is the sky?\nA: The sky is blue.",
    "Q: What is the boiling point of water?\nA: Water boils at 100 degrees Celsius.",
    "Q: How many continents are there?\nA: There are seven continents.",
    "Q: What is the largest planet?\nA: Jupiter is the largest planet.",
    "Q: What is the chemical symbol for gold?\nA: The chemical symbol for gold is Au.",
    "Q: How many days are in a week?\nA: There are seven days in a week.",
    "Q: What is the freezing point of water?\nA: Water freezes at 0 degrees Celsius.",
    "Q: What is the speed of light?\nA: The speed of light is about 300,000 km per second.",
    "Q: Who painted the Mona Lisa?\nA: Leonardo da Vinci painted the Mona Lisa.",
]
dataset = Dataset.from_dict({"text": SAMPLE_TEXTS})

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    dtype="auto",
    device_map="auto",
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Smoke-test config: optimized for speed + minimum OOM risk, not quality ---
sft_config = SFTConfig(
    output_dir="./qwen2.5-3b-lora-adapter",
    max_steps=10,                        # fixed step count, not epochs — this is a smoke test
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,       # small effective batch — fast, low memory
    gradient_checkpointing=True,
    max_length=128,                      # short sequences for the first run
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",                  # saved manually below, once
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)

trainer.train()

trainer.model.save_pretrained("./qwen2.5-3b-lora-adapter")
tokenizer.save_pretrained("./qwen2.5-3b-lora-adapter")

if torch.cuda.is_available():
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n[Measured] Peak-ish GPU memory reserved during training: {reserved:.2f} GB")

print("\nSmoke test complete. Adapter saved to ./qwen2.5-3b-lora-adapter")
print("This proves the pipeline works. It is NOT a meaningful fine-tune — see real-training settings for that.")