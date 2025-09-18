# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # LoRA Training Script for Causal Language Models
# This script demonstrates how to perform LoRA (Low-Rank Adaptation) fine-tuning on a causal language model using the Hugging Face ecosystem [[4]].
# LoRA significantly reduces the number of trainable parameters by freezing the pre-trained model weights and injecting trainable low-rank matrices (adapters) [[9]].

# ## 1. Configuration
# Define training parameters, LoRA configuration, and dataset/model paths.

# %%
# Configuration Dictionary
config = {
    # --- Model Configuration ---
    "model_name": "EleutherAI/gpt-neo-125M", # Example model name
    "dataset_path": "path/to/your/dataset.json", # Path to your instruction dataset (JSON format)
    "output_dir": "./lora-finetuned-model", # Directory to save the trained model

    # --- LoRA Configuration ---
    # LoRA ranks (r) and alpha are key hyperparameters [[5]].
    "lora_r": 8,        # LoRA attention dimension (rank)
    "lora_alpha": 16,   # Scaling factor for LoRA adapters
    "lora_dropout": 0.1, # Dropout probability for LoRA layers
    "lora_target_modules": ["q_proj", "v_proj"], # Modules to apply LoRA to (example for GPT-like models)

    # --- Training Hyperparameters ---
    "epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4, # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 2, # Limit the total amount of checkpoints

    # --- Data Formatting ---
    # Define the chat template structure for your dataset
    "system_prompt": "You are a helpful AI assistant.",
    "user_tag": "### Instruction:",
    "assistant_tag": "### Response:",
    "end_of_turn": "\n",
    "dataset_format": "instruction", # Or 'chat' if your data is in a conversation list format

    # --- Miscellaneous ---
    "seed": 42,
    "fp16": True, # Use mixed precision training if available (requires compatible GPU)
}

print("Configuration loaded:")
for key, value in config.items():
    print(f"  {key}: {value}")


# ## 2. Install Required Packages
# Ensure necessary libraries are installed.

# %%
# !pip install -q transformers datasets accelerate peft trl


# ## 3. Imports
# Import required libraries.

# %%
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # If using quantization
from trl import SFTTrainer # Supervised Fine-Tuning Trainer
import os


# ## 4. Setup and Data Loading
# Initialize device, load tokenizer, model, and dataset.

# %%
# Set seed for reproducibility
set_seed(config["seed"])

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

# --- Load Tokenizer ---
print(f"📥 Loading tokenizer for '{config['model_name']}'...")
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load Dataset ---
print(f"📂 Loading dataset from '{config['dataset_path']}'...")
# Assuming the dataset is a JSON file with a list of examples
# Each example might look like:
# {"instruction": "Summarize the following text...", "input": "...", "output": "..."}
# Or for chat format:
# {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# Load using Hugging Face Datasets library [[11]]
raw_datasets = load_dataset('json', data_files=config['dataset_path'], split='train')
print(f"📊 Dataset loaded with {len(raw_datasets)} examples.")

# --- Format Dataset ---
def format_example(example):
    """Formats a single example according to the chat template."""
    if config["dataset_format"] == "instruction":
        # Combine instruction and input if both exist
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Basic prompt formatting (adjust based on your desired template)
        full_prompt = f"{config['system_prompt']}{config['end_of_turn']}"
        full_prompt += f"{config['user_tag']} {instruction} {input_text}{config['end_of_turn']}"
        full_prompt += f"{config['assistant_tag']} {output_text}{tokenizer.eos_token}"
        return {"text": full_prompt}

    elif config["dataset_format"] == "chat":
        # Assumes 'messages' is a list of dicts with 'role' and 'content'
        messages = example.get("messages", [])
        formatted_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                formatted_text += f"{content}{config['end_of_turn']}"
            elif role == "user":
                formatted_text += f"{config['user_tag']} {content}{config['end_of_turn']}"
            elif role == "assistant":
                formatted_text += f"{config['assistant_tag']} {content}{config['end_of_turn']}"
        formatted_text += f"{config['assistant_tag']}" # Add the start of the assistant's next turn for generation
        return {"text": formatted_text}
    else:
        raise ValueError(f"Unsupported dataset format: {config['dataset_format']}")

print("Formatting dataset according to template...")
# Apply formatting to the entire dataset
lm_datasets = raw_datasets.map(format_example)

# --- Tokenize Dataset ---
def tokenize_function(example):
    """Tokenizes the formatted text."""
    return tokenizer(example["text"], truncation=True, padding=False, max_length=tokenizer.model_max_length)

print("Tokenizing dataset...")
tokenized_datasets = lm_datasets.map(tokenize_function, batched=True, remove_columns=lm_datasets.column_names)
print("✅ Dataset preparation complete.")


# ## 5. Model Loading and LoRA Setup
# Load the base model and apply LoRA adapters.

# %%
# --- Load Model ---
print(f"🧠 Loading model '{config['model_name']}'...")
# Consider using `load_in_8bit=True` or `load_in_4bit=True` with `bitsandbytes` for lower memory usage
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.float16 if config["fp16"] and device.type == "cuda" else torch.float32,
    # device_map="auto", # Automatically distribute model across available devices
    # load_in_8bit=True, # Example for 8-bit quantization (requires bitsandbytes)
)
model = model.to(device)
print(f"✅ Model loaded on {next(model.parameters()).device}")

# --- Prepare Model for LoRA ---
# If using quantization (load_in_8bit/load_in_4bit), prepare the model first
# model = prepare_model_for_kbit_training(model)

# --- Configure LoRA ---
lora_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    target_modules=config["lora_target_modules"],
    lora_dropout=config["lora_dropout"],
    bias="none", # Bias type for Lora. Can be 'none', 'all' or 'lora_only'
    task_type="CAUSAL_LM", # Task type for the model [[5]]
)
print("🔧 Applying LoRA configuration...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Print the number of trainable parameters


# ## 6. Training Setup and Execution
# Define training arguments and start the training process using `SFTTrainer`.

# %%
# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    warmup_steps=config["warmup_steps"],
    logging_steps=config["logging_steps"],
    save_steps=config["save_steps"],
    save_total_limit=config["save_total_limit"],
    fp16=config["fp16"] and device.type == "cuda", # Enable mixed precision if specified and CUDA available
    dataloader_pin_memory=False, # Can help with memory issues
    # report_to="tensorboard", # Enable logging to TensorBoard
    # logging_dir=f"{config['output_dir']}/logs",
    # push_to_hub=False, # Set to True if you want to push the model to Hugging Face Hub
    # hub_model_id="your-username/your-model-name", # Required if push_to_hub=True
)

# --- Data Collator ---
# This handles batching and padding for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Initialize Trainer ---
print("🏋️ Initializing SFT Trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets, # Use the tokenized dataset
    dataset_text_field="text", # Specify the field containing the text in the dataset
    # formatting_func=format_example, # Optional: if you want the trainer to handle formatting internally (requires specific dataset format)
    data_collator=data_collator,
    max_seq_length=tokenizer.model_max_length, # Or a specific length
    # packing=False, # Set to True for packing sequences (can improve efficiency)
)

# --- Start Training ---
print("🚀 Starting training...")
trainer.train()

# --- Save Final Model ---
print(f"💾 Saving final model to '{config['output_dir']}'...")
trainer.save_model(config["output_dir"]) # This saves the LoRA adapters and config
tokenizer.save_pretrained(config["output_dir"]) # Save the tokenizer as well
print("🎉 Training complete!")


# ## 7. (Optional) Merging LoRA Adapters
# Merge the trained LoRA weights back into the base model for easier deployment.

# %%
# from peft import PeftModel
# # Load the base model (ensure it's the same as used for training)
# base_model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float16, low_cpu_mem_usage=True)
# # Load the LoRA adapters
# model = PeftModel.from_pretrained(base_model, config["output_dir"])
# # Merge weights
# model = model.merge_and_unload()
# # Save the merged model
# merged_model_path = f"{config['output_dir']}-merged"
# model.save_pretrained(merged_model_path)
# tokenizer.save_pretrained(merged_model_path)
# print(f"🔗 LoRA adapters merged. Merged model saved to '{merged_model_path}'")

