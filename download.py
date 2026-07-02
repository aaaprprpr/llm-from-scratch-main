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



from datasets import load_dataset
ds = load_dataset("llamafactory/alpaca_zh")
save_path = "./alpaca_zh_local"
print(f"正在导出数据集到: {os.path.abspath(save_path)}")
ds.save_to_disk(save_path)