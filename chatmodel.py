# ============================================================
# Hugging Face Chat Template — Interactive with History
# ============================================================

# Install required packages (quietly, only once)
# !pip install -q transformers accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
model_name = "EleutherAI/gpt-neo-125M"  # change to bigger model if VRAM allows
system_prompt = "You are a helpful AI assistant. Answer clearly and politely."

print(f"📥 Loading tokenizer for '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"🧠 Loading model '{model_name}' in float16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
)

model = model.to(device)
print(f"✅ Model moved to {next(model.parameters()).device}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ------------------------------
# Pipeline
# ------------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type == "cuda" else -1
)

# ============================================================
# Chat System with History
# ============================================================
chat_history = []

def build_prompt(user_message: str) -> str:
    """
    Build the prompt including system message + chat history.
    """
    full_prompt = system_prompt + "\n\n"
    for role, msg in chat_history:
        full_prompt += f"{role}: {msg}\n"
    full_prompt += f"User: {user_message}\nAssistant:"
    return full_prompt

def chat(user_message: str,
         max_new_tokens=200,
         temperature=0.8,
         top_k=50,
         top_p=0.9,
         repetition_penalty=1.2,
         num_beams=1,
         no_repeat_ngram_size=3):
    """
    Run one round of chat with history and save conversation.
    """
    global chat_history

    prompt = build_prompt(user_message)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": True,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.eos_token_id
    }

    outputs = generator(prompt, **gen_kwargs)
    response = outputs[0]["generated_text"][len(prompt):].strip()

    # Save to history
    chat_history.append(("User", user_message))
    chat_history.append(("Assistant", response))

    print(f"\n👤 User: {user_message}")
    print(f"🤖 Assistant: {response}\n")
    return response

# ============================================================
# Example Interactive Chat Loop (run in a cell)
# ============================================================
"""
Run this cell in notebook to chat interactively:

while True:
    msg = input("You: ")
    if msg.lower() in ["exit", "quit"]:
        break
    chat(msg)
"""

# ============================================================
# Saving Chat History (JSON / TXT)
# ============================================================
import json

def save_chat(filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(chat_history, f, indent=2)
    print(f"💾 Chat history saved to {filename}")

def load_chat(filename="chat_history.json"):
    global chat_history
    with open(filename, "r") as f:
        chat_history = json.load(f)
    print(f"📂 Chat history loaded from {filename}")

