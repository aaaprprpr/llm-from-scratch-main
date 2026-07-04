# 项目问题台账

> 审查日期：2026-07-03  
> 审查范围：BPE、数据预处理、预训练、模型实现、HF 转换、SFT、推理、脚本与依赖。  
> 本文只记录问题，不代表所有项目都应立即修改。优先修复顺序以严重度和对现有权重的影响为准。

## 状态约定

- `TODO`：尚未处理。
- `DOING`：正在处理。
- `FIXED`：已经修复并验证。
- `WONTFIX`：确认是有意设计或暂不处理，需补充原因。
- `NEEDS-RETRAIN`：修复会改变 tokenizer、数据流或训练结果，现有权重不能直接继承。

严重度：

- `P0`：直接使训练或推理结果失真。
- `P1`：明显的正确性、兼容性或可复现性问题。
- `P2`：局部错误、脆弱实现或容易踩坑的工程问题。
- `P3`：清理项、性能问题、文档偏差或设计建议。

## 已验证的基线事实

- 当前模型参数量为 **58,733,056**，属于约 59M 参数的教学型语言模型。
- 原始预训练 checkpoint 转成 `main/my_hf_model/model.safetensors` 后，共 87 个 tensor；key、shape 和数值完全一致，最大绝对误差为 **0.0**。因此“权重名称加 `transformer.` 前缀”的格式对齐本身是成功的。
- SFT 数据共 51,155 条，字段为 `instruction/input/output`。
- 全量检查 SFT 的 `encode(prompt)` 与 `encode(prompt + response)`：当前 tokenizer 下 **0 条边界错位**。
- 只有 57 条样本超过 512 token；其中 3 条 prompt 已经占满窗口，当前 filter 会将其排除。
- SFT 的 label shift、prompt `-100` mask 和 EOS label 基本正确，不是当前失败的首要原因。
- `git status` 在本次审查开始和结束时均为空；审查阶段没有修改训练代码。

## P0：必须优先处理

### AUDIT-001：train/val 共用一个全局数据游标

- 状态：`TODO`
- 文件：`main/train_model.py:47-74`、`main/run_train_model.py:64-74`、`main/run_train_model.py:114-178`
- 现象：`get_batch()` 使用模块级 `_current_step_pos`。训练、train loss 估计和 val loss 估计全部修改同一个游标。
- 直接后果：val 数据比 train 数据短，val 扫到末尾时会把游标重置为 0；随后 train 也从训练文件开头重新开始。
- 按当前数据和 50,000 step 配置模拟：
  - train 文件约 21.51 亿 token；
  - 梯度更新一共消费 8.192 亿 token；
  - 梯度实际只使用约 38.73 万个不重复的 256-token 窗口，即约 9914 万个不重复 token；
  - 训练游标最远只走到 1.147 亿附近，占 train 流约 5.33%；
  - val 共触发约 10 次全局重置；大量训练窗口被反复使用。
- 旁证：预训练日志末尾 val loss 从约 3.2 连续跳到约 6，并不代表全局验证集突然退化，而是顺序局部窗口和游标重置造成的失真。
- 建议修法：
  1. train 和 val 使用各自独立的 sampler/generator 状态；
  2. 训练 sampler 与评估 sampler 完全隔离；
  3. 验证应固定随机种子并覆盖随机窗口，或顺序完整扫描；
  4. checkpoint 保存并恢复训练 sampler 状态。
- 验证要求：记录前若干 step 的 train/val 起始索引；验证调用前后，train 下一批索引不得变化。

### AUDIT-002：HF 首次 prefill 会把完整 prompt 裁成最后一个 token

- 状态：`TODO`
- 文件：`main/hf_wrapper.py:119-151`
- 原因：`prepare_inputs_for_generation()` 只判断 `past_key_values is not None`。新版 HF 首次生成会传入一个“非 None、但长度为 0”的 `DynamicCache`，代码因此提前执行 `input_ids = input_ids[:, -1:]`。
- 结果：模型首次 prefill 根本看不到 prompt，只看到最后一个 token。
- 已验证现象：同一个正确中文 prompt：
  - 直接 `forward()` 时，SFT 模型首 token 为“美”，概率约 0.75；
  - `model.generate(use_cache=True)` 首 token 却是 `<|endoftext|>`；
  - `model.generate(use_cache=False)` 能继续生成，证明问题位于 cache/prefill 路径。
- 建议修法：只有 cache 的实际序列长度大于 0 时才裁剪输入；首次空 cache 必须保留完整 prompt。
- 验证要求：对相同 prompt，直接前向的首 token logits 必须与 `generate(max_new_tokens=1)` 完全一致或在浮点误差范围内一致。

## P1：正确性与兼容性问题

### AUDIT-003：DynamicCache 更新分支是死代码

- 状态：`TODO`
- 文件：`main/hf_wrapper.py:97-110`
- 问题：`past_key_values` 是 `forward()` 的显式形参，不会同时出现在 `kwargs` 中。因此 `"past_key_values" in kwargs` 基本不可能成立。
- 结果：代码注释声称会更新 HF `DynamicCache`，实际通常退回自定义 legacy list cache。
- 建议：进入转换逻辑前保存原始 cache 对象，明确支持一种 cache 协议；不要通过 `kwargs` 猜测。

### AUDIT-004：BPE 的 Qwen pattern 被保存成普通字符串，而不是正则

- 状态：`TODO / NEEDS-RETRAIN`
- 文件：`bpe/train_qwen_bpe.py:13-23`
- 问题：`Split(pattern=qwen_pattern, ...)` 最终在 tokenizer JSON 中保存为 `{"String": ...}`，而非 `{"Regex": ...}`。
- 实测：当前 pre-tokenizer 会把整行交给 ByteLevel，当作一个片段；使用真正的 `tokenizers.Regex` 后才会按单词、数字、标点和换行拆分。
- 影响：当前 tokenizer 仍能编码和解码，但并不是代码声称的 Qwen-style 预切分；它可能学到跨空格、跨标点的奇怪 merge。
- 注意：修复后 token ID 和 merge 规则会变化，现有 `.bin`、预训练权重和 SFT 权重全部不兼容。应在下一轮完整重训时处理，不能直接替换当前 tokenizer。

### AUDIT-005：保存的 HF 模型目录不是自包含模型

- 状态：`TODO`
- 文件：`main/hf_wrapper.py:5`、`main/to_hf.py`
- 问题：被复制到 HF 模型目录的 `hf_wrapper.py` 使用绝对导入 `from model import Transformer`，但 `save_pretrained()` 没有把 `model.py` 一同复制进去。
- 结果：模型在当前仓库和特定 `sys.path` 下能加载；目录被移动、上传 Hub 或在干净环境中加载时会找不到 `model`。
- 建议：将模型实现放入可打包模块，并使用相对导入；在一个不包含仓库源码的临时目录做加载测试。

### AUDIT-006：模型完全忽略 attention_mask

- 状态：`TODO`
- 文件：`main/hf_wrapper.py:56-87`、`main/model.py:154-176`、`main/sft.py:75-95`
- 当前情况：SFT 使用右侧 padding，padding labels 为 `-100`。因 causal mask 的方向，右侧 padding 通常不会污染更早的有效 token，所以当前单机 SFT 不是立刻错误。
- 仍然存在的问题：
  - 变长 batch 推理会把 pad 当成真实历史；
  - 左 padding 会直接错误；
  - `pad_token_id == eos_token_id` 时 HF 无法自动推断 mask，并已经给出警告；
  - 不符合通用 HF 模型接口。
- 建议：从 wrapper 一路传递 attention mask 到 SDPA，并正确组合 causal mask 与 padding mask。

### AUDIT-007：SFT 没有验证集，且不同实验共用输出目录

- 状态：`TODO`
- 文件：`main/sft.py:54-105`
- 问题：没有 `eval_dataset`、`eval_strategy`、固定生成样例或最优 checkpoint 选择。
- 现有产物混淆：
  - `checkpoint-1600/2400` 来自较早的 3 epoch 实验；
  - 后来的 1 epoch 实验覆盖了 `checkpoint-800`；
  - `my_sft_complete_model` 对应后来的一轮，而不是目录中 step 最大的 2400。
- 建议：每次运行使用时间戳/实验名目录；保存配置快照；增加独立验证拆分和固定 generation regression cases。

### AUDIT-008：Trainer loss 日志约放大了 gradient accumulation 倍数

- 状态：`TODO`
- 文件：`main/hf_wrapper.py:56-94`、`main/sft.py:63-73`
- 原因：wrapper 有 `**kwargs`，Trainer 因此认为模型会消费 `num_items_in_batch`；实际 wrapper 忽略它并返回当前 micro-batch 的 mean loss。
- 在 `gradient_accumulation_steps=2` 时：Trainer 日志按 optimizer step 累加两个 micro-batch mean，因此 8.2 大致应读作 4.1。
- 梯度说明：Accelerate 的 backward 仍会除以 accumulation steps，因此整体梯度没有简单放大 2 倍；问题主要是日志失真，以及不同 micro-batch 有效 token 数不同时采用等权平均，而不是精确 token 加权。
- 建议：显式声明不接受 loss kwargs，或正确使用 `num_items_in_batch` 计算 summed loss / accumulated valid-token count。

### AUDIT-009：checkpoint 保存时机和 resume 语义不正确

- 状态：`TODO`
- 文件：`main/train_model.py:76-92`、`main/run_train_model.py:103-107`、`main/run_train_model.py:124-188`
- 问题：
  - checkpoint 在当前 iteration 的 forward/backward/step 之前保存；
  - `ckpt_iter_49999.pt` 不包含 iteration 49999 的更新；
  - 注释却声称循环中最后一步已经保存；
  - resume 从保存的 iteration 本身重新开始，会重复一次 iteration；
  - 没有保存 Python/NumPy/Torch/CUDA RNG；
  - 没有保存数据 sampler/游标状态；
  - `torch.load()` 没有统一 `map_location` 策略。
- 建议：optimizer step 后保存 `next_iteration`；完整保存 RNG、scheduler 和 sampler 状态；增加中断恢复一致性测试。

### AUDIT-010：有 past cache 且一次输入多个新 token 时 causal mask 错位

- 状态：`TODO`
- 文件：`main/model.py:168-176`
- 原因：`scaled_dot_product_attention(..., is_causal=True)` 在 query 长度 `T` 小于 key 长度 `past_len + T` 时使用的默认因果对齐并不表示“所有 past + 当前 chunk 的历史”。
- 当前单 token decode 因 `T == 1` 且关闭 causal mask，碰巧可以看到全部 key；多 token chunk decode 会错。
- 建议：存在 past 时显式构造带 past offset 的 causal mask，或严格限制 API 只接受单 token decode 并断言。

### AUDIT-011：HF config 与实际模型能力不一致

- 状态：`TODO`
- 文件：`main/hf_wrapper.py:8-54`
- 问题：
  - config 默认可能声明 `tie_word_embeddings=True`，实际 embedding 与 lm_head 未绑定；
  - 使用 `_all_tied_weights_keys`/property 绕过版本问题，但没有从语义上修正 config；
  - wrapper 构造后没有调用标准 `post_init()`；
  - 没有实现 `get_input_embeddings()`、`get_output_embeddings()` 等常见接口；
  - config 没有可靠设置 bos/eos/pad token id；
  - tokenizer 没有和模型目录一起保存；
  - `context_length` 不是 HF 通常识别的 `max_position_embeddings`。
- 影响：当前加载和简单训练可用，但 resize embeddings、pipeline、Hub、批量 generate 等生态能力不可靠。

## P2：局部 bug 与脆弱实现

### AUDIT-012：`play_model.py` 解包不存在的第二个返回值

- 状态：`TODO`
- 文件：`main/play_model.py:50`、`main/model.py:306`
- 问题：`full_output, _ = model.generate(...)`，但 `generate()` 只返回 `idx`。
- 结果：运行到这里会抛出 unpack 错误。

### AUDIT-013：自定义 generate 不支持 batch EOS 判断

- 状态：`TODO`
- 文件：`main/model.py:299-304`
- 问题：使用 `idx_next.item()`，batch size 大于 1 时会报错；也没有按样本维护 finished mask。

### AUDIT-014：自定义 generate 永久把模型切到 eval 模式

- 状态：`TODO`
- 文件：`main/model.py:249-306`
- 问题：进入时调用 `self.eval()`，退出时不恢复原训练状态。
- 当前模型没有 dropout，所以数值影响有限；未来加入 dropout 后会造成隐蔽错误。

### AUDIT-015：context_length 只检查当前 chunk，不检查 cache 总长度

- 状态：`TODO`
- 文件：`main/model.py:219-246`、`main/model.py:249-306`
- 问题：只断言 `T <= context_length`。进入增量生成后 `T=1`，cache 可以无限增长，超过训练窗口而不报错，也没有 sliding window。
- 相关问题：SFT 把 config 从 256 直接改为 512，依赖 RoPE 外推；数据中真正超过 256 的样本比例很低，256-512 区间没有得到充分训练。

### AUDIT-016：模型参数缺少集中合法性校验

- 状态：`TODO`
- 文件：`main/model.py:136-152`、`main/model.py:199-218`
- 需要校验：`d_model % n_head == 0`、head dim 为偶数、vocab size 与 tokenizer 一致、token id 不越界、context length 为正等。
- 当前部分错误会在 `view()` 或 RoPE assert 中晚发现，报错位置不友好。

### AUDIT-017：训练与验证指标不是稳定、可比较的估计

- 状态：`TODO`
- 文件：`main/run_train_model.py:63-74`
- 除 AUDIT-001 的共享游标外，评估只取当前游标后的连续窗口，没有固定评估索引，也没有全局随机采样。
- 不同 iteration 的 loss 可能对应完全不同来源/主题区间，曲线中的尖峰无法直接解释为模型退化。

### AUDIT-018：顺序训练 sampler 没有 shuffle，恢复训练也不恢复数据位置

- 状态：`TODO`
- 文件：`main/train_model.py:49-74`
- 顺序扫描本身可以是设计选择，但当前数据顺序依赖上游文件和分块顺序；加上 resume 重置，会产生严重的数据分布偏差。
- 建议至少支持确定性的 epoch shuffle 或独立可恢复 sampler。

### AUDIT-019：预训练生成/保存发生在训练 step 之前

- 状态：`TODO`
- 文件：`main/run_train_model.py:146-175`
- 除 checkpoint 少一步外，日志中“iter N 的生成结果”实际来自执行 iteration N 更新之前的模型，容易误读。

### AUDIT-020：RoPE cache 每层重复保存且设备迁移逻辑较绕

- 状态：`TODO`
- 文件：`main/model.py:13-116`
- 每个 attention layer 各自维护相同的 cos/sin cache，浪费少量显存并重复扩容。
- `self.device` 未使用；注释称 einsum 涉及 float64，但实际 positions 和 inv_freq 都已是 float32。
- 这不是当前正确性瓶颈，可后置处理。

### AUDIT-021：Tokenizer 包装器存在意外输出和无用参数

- 状态：`TODO`
- 文件：`main/tokenizer_optimized.py`
- `tokenize()` 会先 print 再返回；库函数不应产生隐式输出。
- `text(idx, device)` 的 `device` 参数完全未使用。
- 没有暴露 `__len__`、标准 special token 属性和 batch encode，导致其他代码反复访问内部 tokenizer。

### AUDIT-022：SFT 脚本在 import 阶段执行大量副作用

- 状态：`TODO`
- 文件：`main/sft.py:6-60`
- 模型加载、数据加载、map 和 filter 都在 `if __name__ == '__main__'` 之外。
- 结果：导入该模块就会加载数百 MB 权重并预处理整个数据集；Windows 多进程或测试导入时尤其危险。
- 建议：所有执行逻辑放入 `main()`，模块顶层只保留定义。

### AUDIT-023：项目入口严重依赖当前工作目录

- 状态：`TODO`
- 涉及：`main/sft.py`、`main/to_hf.py`、`main/test.py`、`main/play_model.py`、`bpe/train_qwen_bpe.py` 等。
- 现状：有的脚本要求从仓库根目录执行，有的要求先 `cd main`，路径字符串同时混用 `/`、`\`、`../`。
- 建议：统一用 `Path(__file__).resolve()` 推导项目根目录，或全部改为 CLI 参数。

## P2：数据处理问题

### AUDIT-024：EOS token id 被硬编码为 0

- 状态：`TODO`
- 文件：`prepare_data.py:8-10`
- 当前 tokenizer 的 `<|endoftext|>` 确实是 0，因此现有数据正确；换 tokenizer 后会静默写错。
- 建议从 tokenizer 查询并断言。

### AUDIT-025：`split_data.py` 创建进程池但使用同步 `pool.apply`

- 状态：`TODO`
- 文件：`split_data.py:22-59`
- 每批任务都阻塞等待，实际上没有并行收益；同时创建 `cpu_count()` 个进程增加额外开销。

### AUDIT-026：train/val 划分未固定随机种子

- 状态：`TODO`
- 文件：`split_data.py:1-20`
- 相同原始数据无法复现完全一致的 split，也无法稳定比较训练实验。

### AUDIT-027：部分“低内存/多进程”脚本实际会整文件读入内存

- 状态：`TODO`
- 文件：`clean_all_data.py:44-51`、`clean_traditional.py:44-66`
- `readlines()` 和收集全部 `valid_lines` 会对大文件占用大量内存，与注释中的低内存目标不一致。

### AUDIT-028：`prepare_data.py` 的 worker 和进度条配置不真实

- 状态：`TODO`
- 文件：`prepare_data.py:48-81`
- 计算了 `max_workers = cpu_count()-1`，实际却硬编码 `max_workers=6`。
- 进度条按 `CHUNK_LINES * 150` 估算字节，最终强制填满，不能反映真实吞吐或剩余时间。

### AUDIT-029：清洗策略非常宽松，且“繁转简”脚本实际是删除繁体

- 状态：`TODO / DESIGN-DECISION`
- 文件：`clean_all_data.py:13-30`、`clean_traditional.py:11-21`
- 任意含一个中文字符的整行都会保留，因此带 URL、乱码、长英文或模板噪声的行仍会进入训练集。
- `clean_traditional.py` 检测到可转换文本后直接丢弃，而不是转换为简体；会损失大量可用数据。
- 用户已明确数据选择带有个人偏好，因此该项需要先决定数据目标，再修改规则。

### AUDIT-030：BPE 只在 val.txt 上训练

- 状态：`TODO / DESIGN-DECISION / NEEDS-RETRAIN`
- 文件：`bpe/train_qwen_bpe.py:41-44`
- 使用 val 子集训练词表不是绝对错误，但词频只来自较小子集，并且验证数据参与了 tokenizer 统计。
- 若要严格 held-out，应在 split 之前训练 tokenizer，或只使用 train 文本并记录抽样规则。

## P2：脚本、依赖和产物

### AUDIT-031：`run.sh` 当前不可用

- 状态：`TODO`
- 文件：`run.sh`
- 明显问题：
  - `VAL_DATA=  "..."`、`VOCAB= "..."` 是非法 Bash 赋值；
  - 使用 Windows 反斜杠路径；
  - 传入 parser 不支持的 `--tokenizer_merges`；
  - README 却把它作为 Quick Start。

### AUDIT-032：`run.ps1` 声明了日志文件但没有写入

- 状态：`TODO`
- 文件：`run.ps1:8-43`
- `$LOG_FILE` 只用于打印，没有把 Python stdout/stderr tee 或重定向到文件。
- 结果：控制台声称日志位于某路径，实际该文件不会生成。

### AUDIT-033：requirements 与实际环境和源码依赖不一致

- 状态：`TODO`
- 文件：`requirements.txt`
- requirements 声明 `numpy==1.21.6`、`torch==2.7.1+cu118`；当前实际环境为 Python 3.12.7、NumPy 2.4.3、Torch 2.11.0+cu128。
- NumPy 1.21.6 与 Python 3.12 不兼容。
- 源码使用但 requirements 未声明：`ftfy`、OpenCC 对应包。
- CUDA 版 Torch 通常还需要明确安装源；单独的 `+cu118` pin 不一定能从默认 PyPI 安装。
- 审查期间还观察到 `pyarrow` 导入发生过 Windows access violation，环境组合需要重新锁定和验证。

### AUDIT-034：README 与真实实现严重脱节

- 状态：`TODO`
- 文件：`README.md`
- README 声称不使用 `nn.Linear`、手写 RMSNorm、手写 AdamW；实际使用 `nn.Linear`、`nn.RMSNorm`、`torch.optim.AdamW`。
- 项目树、notebook 名称、数据目录和 tokenizer 目录均已过期。
- 架构图片引用 `img/architecture.png`，实际文件是 `main/architecture.png`。
- “高性能”表述缺少 benchmark；预训练当前为 FP32，未使用 AMP、compile、分布式或完整性能测量。

### AUDIT-035：下载脚本是硬编码的一次性脚本

- 状态：`TODO`
- 文件：`download.py`、`download_qwen.py`
- `download.py` 重复 import，模块执行即联网下载并写当前目录，没有 main guard、CLI、revision pin 或数据版本记录。
- `download_qwen.py` 同样没有设备选择、revision pin 和 main guard。

### AUDIT-036：`.gitignore` 有重复和不完整表达

- 状态：`TODO`
- 文件：`.gitignore`
- `main/my_hf_model` 重复两次。
- 只写了 `venv/`，应确认是否明确忽略 `.venv/`；当前仓库可能依赖其他 ignore 规则才没有显示环境目录。
- 大量产物目录依赖逐项硬编码，后续新实验名容易误提交大权重。

## P3：设计与维护建议

### AUDIT-037：缺少自动化测试

- 状态：`TODO`
- 当前 `main/test.py` 是手工生成脚本，不是测试套件。
- 建议的最小测试集：
  1. RoPE shape/dtype/device 与参考实现；
  2. causal attention 不看未来 token；
  3. cache decode logits 与 full forward logits 一致；
  4. HF generate 首 token 与 direct forward 一致；
  5. SFT label mask 边界与 EOS；
  6. checkpoint 中断恢复后下一 step 完全一致；
  7. train/val sampler 状态互不影响；
  8. 模型目录复制到干净位置后仍能加载。

### AUDIT-038：配置散落且硬编码

- 状态：`TODO`
- 模型尺寸、路径、context length、tokenizer、checkpoint 和训练参数散落在 PowerShell、Bash 和多个 Python 文件中。
- 建议：至少统一 argparse + JSON/YAML 配置快照；checkpoint 保存模型 config 和 tokenizer fingerprint。

### AUDIT-039：缺少实验元数据与数据指纹

- 状态：`TODO`
- 当前无法仅凭 checkpoint 确认它使用了哪份 tokenizer JSON、train.bin、清洗规则和 Git commit。
- 建议记录：Git commit、dirty 状态、参数、包版本、CUDA/GPU、tokenizer SHA256、数据文件大小/SHA256、随机种子。

### AUDIT-040：部分架构选择应记录为选择，而不是误当 bug

- 状态：`WONTFIX / DOCUMENT`
- 无 dropout、embedding 与 lm_head 不绑权、仅 MHA 而非 GQA、无 bias、SwiGLU、Pre-RMSNorm 都可以是合理选择。
- 这些会影响参数效率和训练表现，但不属于实现错误。修改前应先明确项目目标是“教学实现”“59M 可用中文模型”还是“HF 兼容实验平台”。

## 建议修复顺序

### 路线 A：先抢救和正确评估现有权重

1. 修复 AUDIT-002、003：HF prefill 与 cache。
2. 加入最小 cache/full-forward 对齐测试。
3. 修复 AUDIT-007、008：SFT 独立输出目录、验证集和正确 loss 统计。
4. 支持 attention mask，并用固定 prompt 比较 base、1 epoch、3 epoch checkpoint。
5. 在修复 cache 前，可临时 `use_cache=False` 做质量诊断，但这不是最终方案。

### 路线 B：准备下一轮从头训练

1. 修复 AUDIT-004：真正的 Regex tokenizer，并重新训练 tokenizer。
2. 重新生成全部 `.bin`，保存 tokenizer/data 指纹。
3. 修复 AUDIT-001、009、017、018：独立可恢复 sampler、可靠验证和 checkpoint。
4. 再处理依赖锁定、统一配置和运行入口。
5. 最后重新预训练，再做 SFT；旧权重不能与新 tokenizer 混用。

## 修复记录模板

处理每项时在对应条目下追加：

```text
修复日期：YYYY-MM-DD
修改 commit：<hash>
修改摘要：
验证命令：
验证结果：
是否需要重训：是/否
遗留风险：
```
