# 数据流水线

`configs/data_pipeline.json` 是唯一配置入口。`sources` 定义原始数据的位置和格式；`download.sources`、`preprocess.sources` 指定各阶段要处理的源；`build_bin.input` 选择要编码的输入。所有相对路径都以仓库根目录解析。当前输入是 MiniMind 完整预训练 JSONL；原始文件和已有 bin 已随目录移动，路径见下方配置。

## 当前 MiniMind 路径

```text
sources.minimind.path                  dataset/data_pipeline/data/downloads/minimind/pretrain_t2t.jsonl
build_bin.tokenizer                    tokenize/tokenizer
build_bin.train_bin                    dataset/data_pipeline/data/minimind_full_8192/train.bin
build_bin.val_bin                      dataset/data_pipeline/data/minimind_full_8192/val.bin
```

从仓库根目录执行：

```bash
python -m dataset.data_pipeline.download   # 本地 JSONL 已存在时核对 SHA256；缺失时按固定 revision 下载
python -m dataset.data_pipeline.build_bin  # 验证 JSONL，再划分、编码并写 bin
```

JSONL 每行必须是含非空 `text` 字符串的对象。构建器只保存行偏移，按批读取原文，不生成清洗副本；记录不截断、不补齐，末尾追加当前分词器的 EOS。源文件的 SHA256 与 `sources.minimind.sha256` 不符时直接报错。已有 bin 或元数据且 `build_bin.overwrite=false` 时，会在扫描大文件前报错。成功写入时使用临时文件，bin 元数据记录原始指纹、划分、词表指纹和实际 token 数。

当前 8K bin 与训练配置已存在。上面的构建命令用于重建或换数据时复现，运行前先选新的输出路径；不要覆盖已训练权重对应的 bin。MiniMind 的 mini 版实验需要同时修改 `sources.minimind.filename`、`path`、`sha256` 和 `build_bin.train_bin` / `val_bin`，并同步修改 `configs/pretrain.json` 的路径。

## 其他 Hugging Face 数据集

`fineweb_edu_zh_4_5`、`finewiki_zh`、`wiki_zh_20231101` 和 `tigerresearch_pretrain_zh` 保留为可选源。它们目前未列入 `download.sources` 或 `preprocess.sources`，不会自动参与训练。要使用时：

1. 把源名加入 `download.sources`，运行 `python -m dataset.data_pipeline.download`。下载后保存为本地 Dataset，并在 `dataset/data_pipeline/dataset_samples/` 生成两条结构样本。
2. 把需要的源名加入 `preprocess.sources`，运行 `python -m dataset.data_pipeline.preprocess`。对应 adapter 提取完整文本并清洗，输出单列 `text` 的 Dataset。
3. 将 `build_bin.input` 改为 `preprocess`，设置新的 bin 路径和匹配的 tokenizer，运行 `python -m dataset.data_pipeline.build_bin`。

`preprocess.sources` 只接收 Hugging Face Dataset；MiniMind JSONL 直接进入通用 bin 构建器。标注平台生成的本地 Dataset 仍可直接把路径写入 `build_bin.input`。预处理不做划分或 token 截断；bin 阶段才按记录随机划分训练和验证。默认 `overwrite=false`，各阶段不会静默替换已有产物。

FineWeb 和 FineWiki 的 `data_files` 固定在中文目标子集，Wikipedia 固定为 `20231101.zh`；下载代码会检查这些范围。原始来源的许可和质量约束见对应数据集页面，使用前需按用途确认。
