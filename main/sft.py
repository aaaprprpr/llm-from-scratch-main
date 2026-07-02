from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForCausalLM, AutoConfig
from tokenizer_optimized import Tokenizer
from datasets import load_dataset, load_from_disk
import torch

vocab_file = "../bpe/tokenizer"
tokenizer = Tokenizer(vocab_file)
eos_id = tokenizer.special_token_to_id.get("<|endoftext|>", 0)

config = AutoConfig.from_pretrained("./my_hf_model", trust_remote_code=True)
config.context_length = 512
model = AutoModelForCausalLM.from_pretrained("./my_hf_model", config=config, trust_remote_code=True)

# 【修改点 1】改用本地加载你刚刚存下来的 Alpaca 数据
dataset = load_from_disk("./alpaca_zh_local")
train_raw_dataset = dataset['train']  # alpaca-zh 默认是 'train' 划分

# 【修改点 2】按照你的代码习惯，写一个 Alpaca 的字段转换函数
def format_alpaca_for_my_model(example):
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    # 拼接 Prompt
    if input_text:
        prompt = f"以下是指令和输入的组合，请给出合适的回答。\n指令：{instruction}\n输入：{input_text}\n答案："
    else:
        prompt = f"以下是指令，请给出合适的回答。\n指令：{instruction}\n答案："
        
    response = f"{output_text}"
    return {"prompt": prompt, "response": response}

def tokenize_sft_function(example, max_length=512):
    # 1. 严格计算纯 Prompt 的 Token 长度
    prompt_ids = tokenizer.encode(example['prompt'])
    
    # 2. 把整句话（Prompt + Response）拼起来，整体送进 encode！
    # 这样可以保证中间的衔接 Token 绝对不会因为单独切词而发生碎裂或错位
    full_text = example['prompt'] + example['response']
    input_ids = tokenizer.encode(full_text)
    input_ids.append(eos_id) # 结尾加上结束符

    # 3. 构造 labels：前面属于 Prompt 的长度全给 -100，后面才是真正要学的 Response
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    # 4. 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
    return {"input_ids": input_ids, "labels": labels}

# 用你原来的 map 和 filter 链条，一个字不差
formatted_dataset = train_raw_dataset.map(format_alpaca_for_my_model)
tokenized_dataset = formatted_dataset.map(
    tokenize_sft_function,
    remove_columns=formatted_dataset.column_names
)
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0 and any(l != -100 for l in x['labels']))

# 用你完全通过过的训练参数
training_args = TrainingArguments(
    output_dir="./my_sft_model",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_steps=10,
    num_train_epochs=1,
    bf16=True,
    fp16=False, 
    save_strategy="epoch",
)

# 沿用你完全没问题的原生 custom_data_collator
def custom_data_collator(features):
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

# 必须包裹在守护入口里，确保 Windows 顺畅
if __name__ == '__main__':
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=custom_data_collator
    )
    trainer.train()
    model.save_pretrained("./my_sft_complete_model")