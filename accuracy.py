import sys
print("1. 进程已启动...", flush=True)
import os
print("2. 基础库 OK...", flush=True)
import pandas as pd
print("3. Pandas OK...", flush=True)
from tqdm import tqdm
print("4. Tqdm OK...", flush=True)
import torch
print("5. Torch OK...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
print("6. Transformers OK...", flush=True)
from datasets import load_from_disk
print("7. Datasets OK...", flush=True)
def evaluate_ceval(model_path, dataset, name, sample_limit=None):
    print(f"\n正在加载模型 【{name}】: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 确保 padding token 正常，生成模式一般靠左或靠右 padding，单条推理直接设置即可
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    eval_data = dataset
    if sample_limit and sample_limit < len(dataset):
        print(f"提示：当前开启抽样评估，抽取前 {sample_limit} 条进行生成式评测。", flush=True)
        eval_data = dataset.select(range(sample_limit))

    correct_count = 0
    total_count = len(eval_data)

    for item in tqdm(eval_data, desc=f"生成式评测中 [{name}]", mininterval=1.0):
        question = item['question']
        a_val, b_val, c_val, d_val = item['A'], item['B'], item['C'], item['D']
        gold_answer = item['answer'].strip().upper()

        # 构造符合问答/微调直觉的 Prompt 引导
        prompt = (
            f"请直接回答以下单项选择题，你只需要输出正确选项的字母（A、B、C 或 D），不要输出其他任何解释。\n\n"
            f"题目：{question}\n"
            f"A. {a_val}\n"
            f"B. {b_val}\n"
            f"C. {c_val}\n"
            f"D. {d_val}\n\n"
            f"正确答案是："
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # max_new_tokens=4 足够让它吐出答案和可能的空格
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False, # 使用 greedy search，确保结果稳定不变
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 只取出模型新生成的 token
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()

        # 清洗生成的文本，只要它吐出的前几个字符里包含了 A/B/C/D 就行
        pred_answer = None
        for char in pred_text:
            if char in ['A', 'B', 'C', 'D']:
                pred_answer = char
                break
        
        if pred_answer == gold_answer:
            correct_count += 1

    acc = correct_count / total_count
    
    # 及时释放显存
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return acc

if __name__ == "__main__":
    print("正在读取本地 C-Eval 数据集...", flush=True)
    try:
        ceval_dict = load_from_disk("./ceval_all_local")
        print("数据集读取成功！", flush=True)
    except Exception as e:
        print(f"数据集读取失败: {e}", flush=True)
        sys.exit(1)
        
    target_dataset = ceval_dict['val'] 
    SAMPLE_LIMIT = 2000 

    MODELS_CONFIG = {
        "Qwen2-0.5B 基座模型": r"D:\model\qwen_model",
        "LoRA (LR=5e-5, Ep=1)": r"D:\model\export\lora1",
        "LoRA (LR=1e-5, Ep=1)": r"D:\model\export\lora2",
        "Full全量微调模型(LR=1e-4, Ep=1)": r"D:\model\full_train_2026-05-20-00-44-02",
        "Full全量微调模型(LR=5e-5, Ep=1)": r"D:\model\full1_train_2026-05-19-08-53-29",
        "Full全量微调模型(LR=1e-6, Ep=1)": r"D:\model\full2_train_2026-05-20-01-16-06" 
    }

    results = []
    for name, path in MODELS_CONFIG.items():
        if not os.path.exists(path):
            print(f"跳过【{name}】，本地路径不存在: {path}", flush=True)
            continue
        try:
            accuracy = evaluate_ceval(path, target_dataset, name, sample_limit=SAMPLE_LIMIT)
            results.append({"模型配置/超参": name, "C-Eval 准确率 (Accuracy)": f"{accuracy:.2%}"})
        except Exception as e:
            print(f"模型 【{name}】 评估失败，报错原因: {e}", flush=True)

    print("\n" + "="*20 + " 📊 最终实验定量评估报告 📊 " + "="*20, flush=True)
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False), flush=True)