from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoModelForCausalLM
# 1. 加载你那完美的标准 HF 模型
model = AutoModelForCausalLM.from_pretrained("./my_hf_model", trust_remote_code=True)

# 2. 准备你的 SFT 训练参数
training_args = TrainingArguments(
    output_dir="./my_sft_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_steps=10,
    num_train_epochs=3,
    fp16=True, # 如果显卡支持可以开启
)

# 3. 把你的数据集用分词器 encode 成包含 input_ids 和 labels 的格式
# 4. 直接喂给 Trainer 开启快乐微调！
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
)
trainer.train()