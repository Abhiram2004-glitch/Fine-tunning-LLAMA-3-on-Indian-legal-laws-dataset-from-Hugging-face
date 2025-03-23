from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
import re
import torch


dataset = load_dataset("jizzu/llama2_indian_law_v3")
df = dataset["train"].to_pandas()


train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)


def format_instruction_response(text):
    match = re.match(r"<s>\[INST\](.*?)\[/INST\](.*)</s>", text, re.DOTALL)
    if match:
        return f"<s>[INST] {match.group(1).strip()} [/INST] {match.group(2).strip()} </s>"
    else:
        return text  

train_df["text"] = train_df["text"].apply(format_instruction_response)
val_df["text"] = val_df["text"].apply(format_instruction_response)

train_dataset = Dataset.from_pandas(train_df[["text"]], preserve_index=False)
val_dataset = Dataset.from_pandas(val_df[["text"]], preserve_index=False)


model_name = "unsloth/Llama-3.2-1B-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
  

special_tokens_dict = {
    "additional_special_tokens": ["<s>", "[INST]", "[/INST]", "</s>"]
}
tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.padding_side = "right" 
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)


model.resize_token_embeddings(len(tokenizer))


lora_config = LoraConfig(
    r=16,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none"
)


model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

def preprocess_function(examples):

    examples["text"] = [text.replace("\u200b", "") for text in examples["text"]]

    model_inputs = tokenizer(
        examples["text"],
        max_length=512,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
    )
    
    model_inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in model_inputs["input_ids"]
    ]

    return model_inputs


tokenized_datasets = {
    "train": train_dataset.map(preprocess_function, batched=True),
    "validation": val_dataset.map(preprocess_function, batched=True),
}

# print("Tokenized Example:")
# print(tokenized_datasets["train"][4])

decoded_example = tokenizer.decode(tokenized_datasets["train"][4]["input_ids"])
print("\nDecoded Tokenized Example:")
print(decoded_example)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  
    pad_to_multiple_of=16, 
   
)


training_args = TrainingArguments(
    output_dir="./fine-tuned-llm",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,  
    learning_rate=3e-4,  
    lr_scheduler_type="cosine",  
    max_grad_norm=0.5,
    gradient_checkpointing=False,
    save_total_limit=2,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    
)


trainer.train()


model.save_pretrained("./fine-tuned-llm")
tokenizer.save_pretrained("./fine-tuned-llm")
