# this whole Script is only on GoogleCollab cant be ran inside local terminals due to code compatibilaty

# Cell 1: Install the Magic Libraries
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# to train O-Llama we take 4 main steps

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
import shutil
from google.colab import drive

# step 1: we get the model and tokenizer by quantazizing the model to a 4bit float instead of 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-instruct-bnb-4bit",
    max_seq_length=2048,
    # so this O-llama model can take up to 8k words but we tell it to not look at more than 2048 to save memory
    dtype=None,
    # its either float 16 or Bfloat16 depending on the ram we have unsloth will pick one for us if we set it to None
    load_in_4bit=True  # this turns the model from 16 to 4 bit float

)

# step 2: we get LoRA

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    # this is Rank ##High rank(it learns many things its used to write a book or math stuff) high such as 64 oe 128
    # Low r Rank ## if for when the data is simple and we only want it to learn simple facts and things low as 8 or 4

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # in the old days, we only targeted q and v.
    # By targeting all of them (the whole list), Unsloth
    # Emakes the model much smarter because we are upgrading its logic centers, not just its attention.

    lora_alpha=16,
    # this is the importance and has an equation importanse of new data = r/Lora_alpha since its 1:1 ratio
    # it takes the new data as important as the data it learned in the pre_training Phase
    # higher means that the new data is more importanat lower makes the data seem less important

    lora_dropout=0,
    # we dont want any drop out so that the training is faster and the data we have is memorization so we want it off

    bias="none",  # better for gpu and faster so that bias dosent interfer with output

    use_gradient_checkpointing=True  # this just makes the model use less memmory so it doesnt crash
)

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# this part above

def alpaca_formatting(example):
    # we take each one from the json file
    instructions = example["instruction"]
    inputs = example["input"]
    outputs = example["output"]
    texts = []

    for i, inp, out in zip(instructions, inputs,
                           outputs):  # here we use zip becuase it groups all three together in one itteration
        text = alpaca_prompt.format(i, inp,
                                    out) + tokenizer.eos_token  # we group them and give the the end of sentence token to stop after we got our answer
        texts.append(text)

    return {"text": texts}  # then we return "text" because SFTTrainer looks for A column named text to learn from


dataset = load_dataset("json", data_files=".json", split="train")

dataset = dataset.map(alpaca_formatting, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # here we tell that it sould only look for a column named text and not anything else
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,  # first  5 steps learns very gently (updates the wheighs really small adjusted )
        max_steps=60,  # basically epochs
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),  # if gpu is t4 then float 16 is used
        bf16=torch.cuda.is_bf16_supported(),  # if the gpu is h100 then B16 float is used
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        # when data is small it could overfit so this helps in the overall overfitting by making the wheighs adjustment small
        # Weight Decay: Stops the model from becoming "obsessive" and ruining its brain.

        lr_scheduler_type="linear",
        # from start in steps 0 to 5 it gets the speed of 2e-4 and then in the end it slows down to prevent overshooting
        # Linear Scheduler: Slows down the learning at the end so you don't overshoot the finish line.
        seed=3407,
        output_dir="outputs"
    ),
)

#pipeLine---------------------------------

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
