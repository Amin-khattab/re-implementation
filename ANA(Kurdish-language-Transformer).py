!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes

# this is ran on GC 

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from google.colab import drive

# 0. CLEAR MEMORY
torch.cuda.empty_cache()
import gc
gc.collect()

# 1. SETTINGS GC error handeling 
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_USE_FUSED_CROSSENTROPY"] = "0"  # <-- DISABLE FUSED CE
os.environ["TORCHDYNAMO_DISABLE"] = "1"             # <-- DISABLE DYNAMO

# 2. LOAD MODEL
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
    max_seq_length = 512,
    load_in_4bit = True,
    dtype = None,
    device_map = "cuda:0",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    lora_alpha = 16,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# 3. DATA PREP
dataset = load_dataset("json", data_files={"train": "kurdish_JSONL.jsonl"}, split="train")

max_samples = min(len(dataset), 15000)
dataset = dataset.shuffle(seed=42).select(range(max_samples))
print(f"📊 Using {len(dataset)} examples")

alpaca_prompt = """### Instruction:
{}

### Response:
{}"""

def formater(examples):
    texts = []
    for i, r in zip(examples["instruction"], examples["response"]):
        text = alpaca_prompt.format(i, r) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

formated_dataset = dataset.map(formater, batched=True)

# 4. TRAINER
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formated_dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    packing = False,
    args = TrainingArguments(
        num_train_epochs = 2,
        logging_steps = 25,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        learning_rate = 2e-4,
        weight_decay = 0.01,
        warmup_ratio = 0.03,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        report_to = "none",
        output_dir = "outputs",
        lr_scheduler_type = "cosine",
        seed = 42,
        dataloader_num_workers = 0,
        dataloader_pin_memory = False,
        save_strategy = "no",
        torch_compile = False,  
    )
)

# 5. EXECUTE
print("🚀 Starting training...")
trainer.train()


#pipeline(same as the Llama i created )---------------------------


print("Starting Training...")
trainer_stats = trainer.train()
print("✅ Done! A LlamaBot is ready.")

# --- RUN THIS CELL TO CHAT ---
from unsloth import FastLanguageModel

# 1. Switch to "Inference Mode" (Makes it faster and stops it from learning)
FastLanguageModel.for_inference(model)

print("🤖 Bot is Online! (Type 'exit' to stop)")
print("-" * 40)

while True:
    # 2. Get your input
    user_question = input("You: ")
    
    # Check if you want to quit
    if user_question.lower() in ["exit", "quit", "stop"]:
        print("👋 Bye!")
        break

    # 3. Format the prompt (The same way we trained it)
    prompt = alpaca_prompt.format(
        user_question, # Instruction
        "",            # Input (Leave blank)
        "",            # Output (Leave blank for the AI to fill)
    )

    # 4. Send to GPU
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    # 5. Generate the Answer
    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    
    # 6. Decode (Turn numbers back to words)
    decoded_output = tokenizer.batch_decode(outputs)[0]
    
    # 7. Clean up the output (Cut off the prompt part)
    # We split the text at "### Response:" and take the second half
    response = decoded_output.split("### Response:\n")[-1]
    
    # Remove the "End of Text" token if it shows up
    response = response.replace("<|end_of_text|>", "").strip()

    # 8. Print the result
    print(f"Bot: {response}")
    print("-" * 40)
