from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoModelForCausalLM,AutoConfig
from tokenizer_optimized import Tokenizer
from datasets import load_dataset,load_from_disk
import torch
vocab_file = "../bpe/tokenizer"
tokenizer = Tokenizer(vocab_file)
eos_id = tokenizer.special_token_to_id.get("<|endoftext|>", 0)
config = AutoConfig.from_pretrained("./my_hf_model", trust_remote_code=True)
config.context_length = 512
model = AutoModelForCausalLM.from_pretrained("./my_hf_model", config=config, trust_remote_code=True)
dataset=load_from_disk("../ceval_all_local")
train_raw_dataset = dataset['test']


def format_example_for_my_model(example):
    prompt = (
        f"以下是单项选择题，请选出正确答案。\n"
        f"题目：{example['question']}\n"
        f"A. {example['A']}\n"
        f"B. {example['B']}\n"
        f"C. {example['C']}\n"
        f"D. {example['D']}\n"
        f"答案："
    )
    
    if example.get('explanation'):
        response = f"{example['answer']}\n解析：{example['explanation']}"
    else:
        response = f"{example['answer']}"
        
    return {"prompt": prompt, "response": response}

def tokenize_sft_function(example, max_length=512):
    prompt_ids = tokenizer.encode(example['prompt'])
    response_ids = tokenizer.encode(example['response'])
    if len(response_ids) == 0:
        return {"input_ids": [], "labels": []}
    response_ids.append(eos_id)
    
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
    return {"input_ids": input_ids, "labels": labels}

formatted_dataset = train_raw_dataset.map(format_example_for_my_model)
tokenized_dataset = formatted_dataset.map(
    tokenize_sft_function,
    remove_columns=formatted_dataset.column_names
)
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0 and any(l != -100 for l in x['labels']))
training_args = TrainingArguments(
    output_dir="./my_sft_model",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_steps=10,
    num_train_epochs=3,
    bf16=True,
    fp16=False, 
    save_strategy="epoch",
)


def custom_data_collator(features):
    """
    features 是一个 list 的字典，每个字典长这样: {'input_ids': [...], 'labels': [...]}
    """

    max_input_len = max(len(feature["input_ids"]) for feature in features)
    
    batch_input_ids = []
    batch_labels = []
    
    for feature in features:
        input_ids = feature["input_ids"]
        labels = feature["labels"]

        remainder = max_input_len - len(input_ids)
        
        padded_input_ids = input_ids + [eos_id] * remainder
        padded_labels = labels + [-100] * remainder
        
        batch_input_ids.append(torch.tensor(padded_input_ids, dtype=torch.long))
        batch_labels.append(torch.tensor(padded_labels, dtype=torch.long))

    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels)
    }
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=custom_data_collator
)
trainer.train()
model.save_pretrained("./my_sft_complete_model")