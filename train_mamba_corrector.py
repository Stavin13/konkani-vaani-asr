import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Point this to your uploaded CSV file in Lightning AI
CSV_PATH = "train_audit.csv" 
MODEL_ID = "state-spaces/mamba-130m-hf"

print(f"Loading tokenizer and model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load the Mamba model (causal LM for text generation)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

# ==========================================
# 2. Data Preparation
# ==========================================
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# We only need the noisy output (hyp_greedy) and the ground truth (ref)
df = df.dropna(subset=['hyp_greedy', 'ref'])

def format_instruction(noisy_text, clean_text):
    # We teach Mamba by giving it a prompt and the expected answer
    return f"Correct this Konkani ASR text: {noisy_text}\nCorrected: {clean_text}{tokenizer.eos_token}"

# Apply formatting
df['prompt'] = df.apply(lambda row: format_instruction(row['hyp_greedy'], row['ref']), axis=1)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['prompt']])

def tokenize_function(examples):
    # Tokenize the combined prompt. The model learns to predict the next token.
    tokens = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=128)
    # For causal LM, labels are the same as input_ids. The loss function ignores padding.
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["prompt"])

# Split into train and validation sets
split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ==========================================
# 3. Training Loop Configuration
# ==========================================
training_args = TrainingArguments(
    output_dir="./mamba-konkani-corrector",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True, # Use mixed precision for fast training on Lightning AI GPUs
    logging_steps=100,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# ==========================================
# 4. Train!
# ==========================================
print("Starting training on Lightning AI...")
trainer.train()

# Save the final corrected model
model.save_pretrained("./mamba-konkani-final")
tokenizer.save_pretrained("./mamba-konkani-final")
print("Training complete! Model saved to ./mamba-konkani-final")