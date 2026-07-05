# 预训练问题台账

> 最近更新：2026-07-03  
> 当前范围：`main/`、`bpe/`、`data_pipeline/`、预训练启动脚本与运行环境。  
> SFT/Hugging Face 转换已经移出当前主线，不在本文继续跟踪。

本文只保留当前预训练链路中仍然存在的问题。已经修复的模型推理问题、已经被新数据流水线替代的旧脚本问题，以及当前范围外的 SFT/HF 问题均已删除。

## 当前结论

- `main/` 现在是纯预训练实现。
- 模型侧的 attention mask、带 cache 的因果遮罩、批量 EOS、生成后训练状态恢复、cache 总长度限制和参数合法性校验已经落地，不再列为问题。
- `data_pipeline/` 已替代原先分散的数据脚本：EOS 查询、固定随机种子、流式清洗、繁简策略、多进程编码和 tokenizer/bin 元数据均已处理。
- 当前最重要的遗留问题仍在训练数据游标、评估方式和 checkpoint/resume 语义。它们会影响训练覆盖率、曲线可信度和断点恢复结果。
- tokenizer 的正则预切分问题仍然存在，但修改后必须重新训练 tokenizer、重新生成 `.bin` 并从头预训练；当前正在训练的模型不能中途切换。

## P0：下轮预训练前必须处理

### PRETRAIN-001：训练与评估共用全局数据游标

- 文件：`main/train_model.py`、`main/run_train_model.py`
- `get_batch()` 使用模块级 `_current_step_pos`；训练 batch、train loss 评估和 val loss 评估都会修改同一个游标。
- val 数据扫到末尾时会把全局游标重置为 0，随后 train 也从训练数据开头重新开始。
- `estimate_loss()` 会消耗训练游标，因此仅仅执行一次评估就会改变下一次真正训练所用的数据。
- 当前顺序采样没有 shuffle，resume 也不恢复数据位置；训练覆盖率和数据分布会进一步偏离预期。
- 现有 loss 曲线来自不同的连续局部窗口，不能稳定比较；尖峰不应直接解释为模型退化。

建议改为：

1. train sampler、train-eval sampler、val sampler 完全独立；
2. 验证使用固定索引或固定随机种子，保证不同 step 可比较；
3. sampler 状态可保存、可恢复；
4. 增加测试，确保调用评估前后下一批训练索引不变。

### PRETRAIN-002：checkpoint 保存时机和 resume 语义不正确

- 文件：`main/train_model.py`、`main/run_train_model.py`
- checkpoint 和生成样例发生在当前 iteration 的 forward/backward/optimizer step 之前。
- `ckpt_iter_N.pt` 实际不包含第 N 次更新，“iter N 的生成结果”也来自更新前模型。
- checkpoint 保存的是当前 `iteration`，resume 又从该 iteration 开始，会重复一次训练 step。
- checkpoint 没有保存 Python、NumPy、Torch、CUDA RNG，也没有保存数据游标/sampler 状态。
- `load_checkpoint()` 没有统一的 `map_location` 策略。
- 循环末尾注释声称最后一步已经保存，但当前保存发生在最后一步更新之前。

建议在 optimizer step 后保存 `next_iteration`，同时保存 RNG、sampler 和必要的训练状态，并用“连续训练 vs 中断恢复”做逐步一致性测试。

## P1：需要下一次完整重训才能处理

### PRETRAIN-003：BPE 的 Qwen pattern 没有作为正则表达式保存

- 文件：`bpe/train_qwen_bpe.py`
- `Split(pattern=qwen_pattern, ...)` 接收的是普通字符串，tokenizer JSON 中会保存为 `String`，而不是 `Regex`。
- 当前 tokenizer 能正常编码和解码，但没有真正执行代码声称的 Qwen-style 预切分，可能产生跨空格或跨标点的异常 merge。
- 需要显式使用 `tokenizers.Regex`，并检查保存后的 tokenizer JSON。

该修改会改变 token ID 和 merge 规则。处理时必须重新训练 tokenizer、重新生成全部 `.bin` 并从头预训练，不能替换当前模型所用 tokenizer。

### PRETRAIN-004：BPE 训练入口仍直接使用 val 文本

- 文件：`bpe/train_qwen_bpe.py`
- 脚本仍硬编码 `../data/val.txt`，与 `data_pipeline/README.md` 中“只使用 train/BPE chunk”的新约定没有接通。
- 如果需要严格 held-out，验证文本不应参与 tokenizer 词频统计。
- BPE 脚本还依赖当前工作目录，输入、输出和训练参数均未提供 CLI。

建议让 BPE 入口显式接收 train 文本或 `bpe/data` chunks，并记录输入文件、抽样规则、参数和 tokenizer SHA256。

## P2：低优先级工程问题

### PRETRAIN-005：运行入口和配置仍不统一

- `run.sh` 仍有非法 Bash 赋值、Windows 反斜杠路径和不存在的 `--tokenizer_merges` 参数，当前不可用。
- `run.ps1` 声明了 `train.log`，但 stdout/stderr 没有写入该文件。
- `main/play_model.py`、BPE 脚本等仍依赖特定当前工作目录。
- 模型尺寸、路径、训练参数和 tokenizer 配置散落在多个脚本中。

建议统一为从仓库根目录执行的 CLI，并让每次运行保存完整参数快照。

### PRETRAIN-006：依赖与 README 仍然失真

- `requirements.txt` 中的 NumPy/Torch pin 与当前 Python/CUDA 环境不匹配，CUDA 版 Torch 也没有说明安装源。
- README 仍声称手写 `nn.Linear`、RMSNorm 和 AdamW，但当前实现使用 PyTorch 官方模块。
- README 的项目树、架构图片路径、Quick Start 和“高性能”描述与当前仓库不一致。

这些问题不影响正在运行的进程，但会影响新环境复现，应在下一次对外使用仓库前处理。

### PRETRAIN-007：Tokenizer 包装器仍有小型接口问题

- 文件：`main/tokenizer_optimized.py`
- `tokenize()` 会隐式打印一次结果后再返回。
- `text(idx, device)` 的 `device` 参数未使用。
- 没有暴露 `__len__` 等常用接口，调用方仍会访问内部 tokenizer 属性。

### PRETRAIN-008：自动化回归测试仍然缺失

最低限度应覆盖：

1. causal attention 不读取未来 token；
2. cache decode logits 与 full forward logits 对齐；
3. attention mask 与多 token chunk cache；
4. batch generation 的 EOS 行为和训练状态恢复；
5. train/val/eval sampler 互不影响；
6. checkpoint 中断恢复后的下一 step 与连续训练一致。

### PRETRAIN-009：checkpoint 的实验元数据仍不完整

`data_pipeline` 已为 tokenizer 和 `.bin` 记录部分指纹，但 checkpoint 本身仍不能确认使用了哪份 tokenizer、数据、清洗规则和代码版本。

建议至少保存 Git commit/dirty 状态、完整参数、包与 CUDA 版本、tokenizer SHA256、数据 meta/SHA256、随机种子和 sampler 状态。

## 当前处理顺序

1. 当前预训练继续运行，不中途更换 tokenizer 或数据格式。
2. 模型产出后做固定 prompt 和固定验证样本评估；解释现有训练曲线时保留 PRETRAIN-001 的限制。
3. 下一轮训练前先修 PRETRAIN-001 和 PRETRAIN-002，并补最小恢复一致性测试。
4. 若决定修 tokenizer，则同时处理 PRETRAIN-003 和 PRETRAIN-004，随后重建全部数据并从头训练。
5. 其余工程清理不阻塞当前模型产出。
