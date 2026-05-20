import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

def get_sentence_ppl(model, tokenizer, sentence):
    """计算单个句子的困惑度 PPL"""
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    inputs["labels"] = inputs["input_ids"].clone()
    
    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs.loss
        ppl = torch.exp(loss).item()
    return ppl

def evaluate_ceval(model_path, dataset, name, sample_limit=None):
    """
    评估单个模型在 C-Eval 上的表现
    sample_limit: 选填。如果 12342 条太长跑不完，可以设为 500 或 1000 做抽样评估
    """
    print(f"\n正在加载模型 【{name}】: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 统一 padding token 避免报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, # Qwen2 推荐用 bf16 跑推理
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 截取评估数据
    eval_data = dataset
    if sample_limit and sample_limit < len(dataset):
        print(f"提示：测试集较大，当前开启抽样评估，抽取前 {sample_limit} 条进行定量评测。")
        eval_data = dataset.select(range(sample_limit))

    correct_count = 0
    total_count = len(eval_data)

    for item in tqdm(eval_data, desc=f"评测中 [{name}]"):
        question = item['question']
        options = {
            'A': item['A'],
            'B': item['B'],
            'C': item['C'],
            'D': item['D']
        }
        gold_answer = item['answer'].strip().upper()

        ppl_results = {}
        # 遍历 A, B, C, D 四个选项计算 PPL
        for label, option_text in options.items():
            # 构造输入给 PPL 模型的句子，这里融入了多选的上下文结构
            full_sentence = f"问题：{question}\n正确答案是：{label}. {option_text}"
            ppl = get_sentence_ppl(model, tokenizer, full_sentence)
            ppl_results[label] = ppl

        # 预测 PPL 最低的选项
        pred_answer = min(ppl_results, key=ppl_results.get)

        if pred_answer == gold_answer:
            correct_count += 1

    acc = correct_count / total_count
    
    # 显存回收，防止循环加载下一个模型时 OOM
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return acc

if __name__ == "__main__":
    from datasets import load_from_disk
    
    # 1. 加载你本地存好的 C-Eval 数据集
    print("正在读取本地 C-Eval 数据集...")
    ceval_dict = load_from_disk("./ceval_all_local")
    
    # 这里我们选择用 val 集（1346条）或者 test 集做评估
    # 注意：学术界测 C-Eval 官方标准一般用 val（因为 test 没给答案需提交官方，但你缝合的如果 test 带 answer 就用 test）
    target_dataset = ceval_dict['val'] 
    
    # 💡 技巧：如果模型太多，1346条跑一次也要很久，可以先设为 300 条快速看超参趋势，出最终报告再设为 None 跑全量
    SAMPLE_LIMIT = 500 

    # 2. 配置你要批量测试的所有模型路径（把你的全量、各种超参 LoRA 路径填进来）
    MODELS_CONFIG = {
        "Qwen2-0.5B 基座模型": r"C:\Users\guojia\PycharmProjects\heima\llm-from-scratch-main\model\qwen_model",
        
        # --- LoRA 消融实验组 ---
        "LoRA (LR=5e-5, Ep=1)": r"C:\Users\guojia\PycharmProjects\heima\llm-from-scratch-main\model\lora_train_2026-05-19-01-09-47",
        "LoRA (LR=1e-5, Ep=1)": r"C:\Users\guojia\PycharmProjects\heima\llm-from-scratch-main\model\lora2_train_2026-05-19-09-39-13",
        
        # --- 全量微调对比组 ---
        "Full全量微调模型": r"C:\Users\guojia\PycharmProjects\heima\llm-from-scratch-main\model\full_train_2026-05-20-00-44-02" ,
        "Full全量微调模型": r"C:\Users\guojia\PycharmProjects\heima\llm-from-scratch-main\model\full1_train_2026-05-19-08-53-29" ,
        "Full全量微调模型": r"C:\Users\guojia\PycharmProjects\heima\llm-from-scratch-main\model\full2_train_2026-05-20-01-16-06" 
    }

    # 3. 循环自动评测
    results = []
    for name, path in MODELS_CONFIG.items():
        if not os.path.exists(path) and "Qwen/" not in path:
            print(f"跳过【{name}】，路径不存在: {path}")
            continue
        try:
            accuracy = evaluate_ceval(path, target_dataset, name, sample_limit=SAMPLE_LIMIT)
            results.append({"模型配置/超参": name, "C-Eval 准确率 (Accuracy)": f"{accuracy:.2%}"})
        except Exception as e:
            print(f"模型 【{name}】 评估失败，报错原因: {e}")

    # 4. 打印最终大作业对比表格
    print("\n" + "="*20 + " 📊 最终实验定量评估报告 📊 " + "="*20)
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))