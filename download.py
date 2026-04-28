from datasets import load_dataset
from tqdm import tqdm  # 进度条



DATASET_NAME = "Orphanage/Baidu_Tieba_KangYaBeiGuo"
OUTPUT_DIR = "data"
OUTPUT_NAME = "Baidu_Tieba_KangYaBeiGuo"
#     "kurehamnm/Chinese_Question_Answering_Dataset"     ,split='train'
#     "SUSTech/ChineseSafe"    ,split='test'
#     "Hanversion/Tieba-SomeInteresting" ,split='train'
#     "Orphanage/Baidu_Tieba_KangYaBeiGuo"  ,split='train'
#     "ticoAg/Belle_train_3.5M_CN"  , split="train"
#     
#     
#     "wikimedia/wikipedia", "20231101.zh", split="train"

print("正在加载数据集...")
dataset = load_dataset(DATASET_NAME,split='train')
print(dataset)
print(dataset[0])



with open( OUTPUT_DIR+'/'+OUTPUT_NAME+'.txt'  ,"w", encoding="utf-8") as f:
    for item in tqdm(dataset, desc="导出文本进度"):
        f.write(item["标题"] + "\n")
        f.write(item["楼主内容"] + "\n")
        for reply in item["回复列表"]:
            f.write(reply + "\n")




