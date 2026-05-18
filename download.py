from datasets import load_dataset
from tqdm import tqdm  # 进度条
import os


# DATASET_NAME = "Orphanage/Baidu_Tieba_KangYaBeiGuo"
# OUTPUT_DIR = "data"
# OUTPUT_NAME = "Baidu_Tieba_KangYaBeiGuo"
# #     "kurehamnm/Chinese_Question_Answering_Dataset"     ,split='train'
# #     "SUSTech/ChineseSafe"    ,split='test'
# #     "Hanversion/Tieba-SomeInteresting" ,split='train'
# #     "Orphanage/Baidu_Tieba_KangYaBeiGuo"  ,split='train'
# #     "ticoAg/Belle_train_3.5M_CN"  , split="train"
# #     
# #     
# #     "wikimedia/wikipedia", "20231101.zh", split="train"
# all=['accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine', 'business_administration', 'chinese_language_and_literature', 'civil_servant', 'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics', 'college_programming', 'computer_architecture', 'computer_network', 'discrete_mathematics', 'education_science', 'electrical_engineer', 'environmental_impact_assessment_engineer', 'fire_engineer', 'high_school_biology', 'high_school_chemistry', 'high_school_chinese', 'high_school_geography', 'high_school_history', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'ideological_and_moral_cultivation', 'law', 'legal_professional', 'logic', 'mao_zedong_thought', 'marxism', 'metrology_engineer', 'middle_school_biology', 'middle_school_chemistry', 'middle_school_geography', 'middle_school_history', 'middle_school_mathematics', 'middle_school_physics', 'middle_school_politics', 'modern_chinese_history', 'operating_system', 'physician', 'plant_protection', 'probability_and_statistics', 'professional_tour_guide', 'sports_science', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine']
# for name in all:
#     print("正在加载数据集...")
#     # dataset = load_dataset(DATASET_NAME,split='train')
#     dataset=load_dataset("ceval/ceval-exam", name)
#     print(dataset)
#     print(dataset['val'][0])



# with open( OUTPUT_DIR+'/'+OUTPUT_NAME+'.txt'  ,"w", encoding="utf-8") as f:
#     for item in tqdm(dataset, desc="导出文本进度"):
#         f.write(item["标题"] + "\n")
#         f.write(item["楼主内容"] + "\n")
#         for reply in item["回复列表"]:
#             f.write(reply + "\n")



import os
from datasets import load_dataset, concatenate_datasets,load_from_disk

# # 1. 把报错信息里提示的所有 52 个学科名字拷贝进来
# SUBJECTS = [
#     'accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine', 
#     'business_administration', 'chinese_language_and_literature', 'civil_servant', 
#     'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics', 
#     'college_programming', 'computer_architecture', 'computer_network', 'discrete_mathematics', 
#     'education_science', 'electrical_engineer', 'environmental_impact_assessment_engineer', 
#     'fire_engineer', 'high_school_biology', 'high_school_chemistry', 'high_school_chinese', 
#     'high_school_geography', 'high_school_history', 'high_school_mathematics', 'high_school_physics', 
#     'high_school_politics', 'ideological_and_moral_cultivation', 'law', 'legal_professional', 
#     'logic', 'mao_zedong_thought', 'marxism', 'metrology_engineer', 'middle_school_biology', 
#     'middle_school_chemistry', 'middle_school_geography', 'middle_school_history', 
#     'middle_school_mathematics', 'middle_school_physics', 'middle_school_politics', 
#     'modern_chinese_history', 'operating_system', 'physician', 'plant_protection', 
#     'probability_and_statistics', 'professional_tour_guide', 'sports_science', 'tax_accountant', 
#     'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine'
# ]

# print(f"🚀 开始循环下载 C-Eval 共 {len(SUBJECTS)} 个学科的全部数据...")

# all_dev_lists = []
# all_val_lists = []
# all_test_lists = []

# for i, sub in enumerate(SUBJECTS):
#     print(f"[{i+1}/{len(SUBJECTS)}] 正在下载: {sub} ...")
#     try:
#         # 逐个子集下载
#         dataset = load_dataset("ceval/ceval-exam", name=sub)
        
#         # 分别把每个学科的子切片存起来
#         all_dev_lists.append(dataset['dev'])
#         all_val_lists.append(dataset['val'])
#         all_test_lists.append(dataset['test'])
#     except Exception as e:
#         print(f"❌ 下载 {sub} 失败，错误信息: {e}")

# print("\n📦 正在进行全学科数据大缝合...")

# # 2. 使用官方推荐的 concatenate_datasets 把所有学科的 Dataset 拼接在一起
# full_dataset_dict = {
#     "dev": concatenate_datasets(all_dev_lists),
#     "val": concatenate_datasets(all_val_lists),
#     "test": concatenate_datasets(all_test_lists)
# }

# from datasets import DatasetDict
# final_dataset = DatasetDict(full_dataset_dict)

# print("\n--- 缝合完成！全量数据集统计 ---")
# print(final_dataset)

# # 3. 强烈建议持久化保存到本地，这样以后 SFT 脚本里直接从本地读，秒加载！
# save_path = "./ceval_all_local"
# final_dataset.save_to_disk(save_path)
# print(f"💾 已经成功把全量 C-Eval 数据集保存到了本地路径: {save_path}")
dataset = load_from_disk("./ceval_all_local")
print(dataset)  # 看看里面到底有哪些 split (train, val, dev) 以及各自的数量
print(dataset['val'][0])
print(dataset['dev'][0])
print(dataset['test'][0])