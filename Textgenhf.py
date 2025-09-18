# ============================================================
# Hugging Face Text Generation with EXPLICIT CUDA (Kaggle/Colab)
# ============================================================

# Install required packages (quietly)
# !pip install -q transformers accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("✅ Libraries imported.")

# ------------------------------
# Device Setup — EXPLICIT CUDA
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# ------------------------------
# Settings
# ------------------------------
model_name = "gpt2"  # Try "EleutherAI/gpt-neo-125M" for more fun, or larger if VRAM allows

# Load tokenizer
print(f"📥 Loading tokenizer for '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in half precision for efficiency
print(f"🧠 Loading model '{model_name}' in float16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Save VRAM
)

# 🔥 EXPLICITLY MOVE MODEL TO CUDA/CPU
model = model.to(device)
print(f"✅ Model moved to {next(model.parameters()).device}")

# Optional: Enable CUDA memory optimizations (if on GPU)
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True  # For Ampere+ GPUs
    torch.backends.cudnn.benchmark = True         # Optimize for fixed input sizes

# ------------------------------
# Text Generation Pipeline
# ------------------------------
print("🔧 Setting up text generation pipeline...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type == "cuda" else -1  # Pipeline still needs device index
)

==================================================================================
# ------------------------------
# Generation Parameters
# ------------------------------
prompt = "In a distant future, humans and AI coexist, and"

gen_kwargs = {
    "max_new_tokens": 200,          # total tokens (prompt + generated)          # minimum tokens to generate
    "do_sample": True,           # enable sampling (for creative outputs)
    "temperature": 0.8,          # randomness (higher = more random)
    "top_k": 50,                 # top-k sampling
    "top_p": 0.9,                # nucleus sampling
    "repetition_penalty": 1.2,   # discourage repetition
    "num_beams": 1,              # beam search (set >1 for beam search)
    "no_repeat_ngram_size": 3,   # block repeating n-grams
    "early_stopping": True,      # stop when EOS reached
    "num_return_sequences": 1,   # generate multiple outputs
    "pad_token_id": tokenizer.eos_token_id  # avoid warnings
}

# ------------------------------
# Run Generation
# ------------------------------
outputs = generator(prompt, **gen_kwargs)

# ------------------------------
# Show Results
# ------------------------------
print("\n" + "="*60)
for i, out in enumerate(outputs):
    print(f"\n=== 🎨 Generated Text {i+1} ===\n")
    print(out["generated_text"])
print("\n" + "="*60)

