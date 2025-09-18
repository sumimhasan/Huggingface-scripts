# ============================================================
# Hugging Face Text Generation with CUDA (Kaggle Notebook)
# ============================================================

!pip install -q transformers accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------------------
# Settings
# ------------------------------
# Choose a model (you can switch to other models from Hugging Face Hub)
model_name = "gpt2"  # Example: "meta-llama/Llama-2-7b-chat-hf" (needs more VRAM)

# Load model & tokenizer with CUDA
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # use float16 for efficiency
    device_map="auto"          # automatically place on GPU if available
)

# ------------------------------
# Text Generation Pipeline
# ------------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# ------------------------------
# Generation Parameters
# ------------------------------
prompt = "In a distant future, humans and AI coexist, and"

gen_kwargs = {
    "max_length": 200,           # total tokens (prompt + generated)
    "min_length": 30,            # minimum tokens to generate
    "do_sample": True,           # enable sampling (for creative outputs)
    "temperature": 0.8,          # randomness (higher = more random)
    "top_k": 50,                 # top-k sampling
    "top_p": 0.9,                # nucleus sampling
    "repetition_penalty": 1.2,   # discourage repetition
    "num_beams": 1,              # beam search (set >1 for beam search)
    "no_repeat_ngram_size": 3,   # block repeating n-grams
    "early_stopping": True,      # stop when EOS reached
    "num_return_sequences": 2,   # generate multiple outputs
    "pad_token_id": tokenizer.eos_token_id  # avoid warnings
}

# ------------------------------
# Run Generation
# ------------------------------
outputs = generator(prompt, **gen_kwargs)

# Show results
for i, out in enumerate(outputs):
    print(f"\n=== Generated Text {i+1} ===\n")
    print(out["generated_text"])
