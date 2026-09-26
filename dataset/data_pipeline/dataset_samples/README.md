# 数据集结构样本

`python -m dataset.data_pipeline.download` 会为选中的源生成 `<source_id>.sample.json`，只保留前两条样本的截断内容，供检查字段和格式。它们不作为训练输入。原始文件路径由 `configs/data_pipeline.json` 的 `sources` 定义。
