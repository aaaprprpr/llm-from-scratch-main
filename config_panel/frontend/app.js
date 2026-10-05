"use strict";

const labels = {
  paths: "文件与路径", model: "模型结构", train: "训练参数", optimizer: "优化器",
  lr_schedule: "学习率调度", logging: "日志与保存", sample: "生成采样", play: "生成预览",
  data: "数据集", sources: "数据源", download: "下载", preprocess: "预处理",
  build_bin: "构建训练 bin", export: "模型导出", hf: "Hugging Face", gguf: "GGUF",
  tokenizer: "分词器", tokenizer_vocab: "分词器路径", train_data: "训练数据", val_data: "验证数据",
  out_root: "训练输出目录", resume: "恢复训练权重", dataset: "数据集路径", chat_template: "对话模板",
  pretrained_weights: "预训练权重", sft_logs: "SFT 输出目录", sft_checkpoint: "SFT 权重",
  tokenized_cache: "分词缓存", dpo_logs: "DPO 输出目录", clean_weights: "纯模型权重",
  hf_export: "Hugging Face 导出目录", vocab_size: "词表大小", context_length: "上下文窗口",
  n_head: "注意力头数", n_kv_head: "KV 头数", num_kv_heads: "KV 头数", theta: "RoPE 基数",
  num_layers: "层数", d_model: "模型宽度", d_ff: "前馈层宽度", tie_word_embeddings: "共享词嵌入权重",
  device: "运行设备", seed: "随机种子", sequence_length: "训练序列长度", batch_size: "批大小",
  tokens_per_update: "每次更新的 token 数", precision: "数值精度", activation_checkpointing: "激活检查点",
  require_flash_attention: "要求 FlashAttention", eval_interval: "验证间隔", eval_tokens: "验证 token 数",
  eval_seed: "验证随机种子", log_interval: "日志间隔", type: "类型", weight_decay: "权重衰减",
  beta1: "一阶动量 β₁", beta2: "二阶动量 β₂", eps: "数值稳定项 ε", fused: "融合优化器",
  max_norm: "梯度裁剪阈值", muon: "Muon 参数", max_lr: "最大学习率", min_lr: "最小学习率",
  momentum: "动量", nesterov: "Nesterov 动量", ns_steps: "正交化迭代次数", adjust_lr_fn: "学习率调整方式",
  warmup_iters: "预热步数", use_wandb: "使用 W&B", wandb_project: "W&B 项目名称",
  checkpoint_interval_multiplier: "权重保存间隔倍数", prompt: "输入提示", max_new_tokens: "最大生成 token 数",
  temperature: "采样温度", top_p: "Top-p", prompts: "预览提示列表", adapter: "数据适配器",
  dataset_id: "数据集 ID", subsets: "数据子集", split: "数据划分", revision: "版本",
  deduplicate: "去重", tokenize_workers: "分词进程数", tokenize_chunksize: "分词任务块大小",
  num_workers: "加载进程数", max_seq_len: "最大序列长度", micro_batch_size: "单步批大小",
  gradient_accumulation_steps: "梯度累积步数", num_epochs: "训练轮数", max_learning_rate: "最大学习率",
  min_learning_rate: "最小学习率", warmup_steps: "预热步数", lr_decay_steps: "学习率衰减步数",
  grad_clip: "梯度裁剪阈值", end_token_weight: "结束 token 权重", eval_every: "验证间隔",
  save_every: "保存间隔", eval_batches: "验证批次数", val_size: "验证集大小",
  length_bucket_multiplier: "长度分桶倍数", beta: "DPO β", average_logprob: "平均序列对数概率",
  kind: "来源类型", repo: "仓库 ID", filename: "文件名", sha256: "文件摘要", path: "路径",
  data_files: "数据文件", config: "数据集配置", cleanup_cache: "清理下载缓存", input: "输入来源",
  output: "输出目录", fix_text: "修正文本文字", max_repetition_ratio: "最大重复比例",
  workers: "处理进程数", overwrite: "覆盖已有输出", train_bin: "训练 bin 路径",
  val_bin: "验证 bin 路径", train_ratio: "训练集比例", backend: "实现后端",
  qk_norm: "Q/K 归一化", dropout: "Dropout", max_steps: "最大训练步数", max_iters: "最大训练步数",
  checkpoint: "权重路径", output_dir: "输出目录", dtype: "权重精度", format: "导出格式",
  llm_cleaning: "模型辅助清洗", provider: "模型服务", base_url: "服务地址",
  context_tokens: "服务上下文长度", max_output_tokens: "最大输出 token 数", timeout_seconds: "请求超时（秒）",
  max_document_characters: "最大文档字符数", max_chunks: "最多分块数", max_units_per_chunk: "每块最大单元数",
  max_parallel_chunks: "并行块数", max_chunk_characters: "每块最大字符数", max_attempts_per_chunk: "每块尝试次数",
  content_risk_fallback: "备用清洗服务"
};

const hints = {
  context_length: "模型允许的位置长度上限。", sequence_length: "预训练每条样本实际使用的序列长度。",
  tokens_per_update: "与批大小、序列长度共同决定梯度累积次数。",
  micro_batch_size: "一次前向与反向计算的样本数。", gradient_accumulation_steps: "累积这些步之后更新一次权重。",
  resume: "未设置时从训练入口指定的起点开始；填写权重路径以恢复。",
  pretrained_weights: "SFT 的初始权重。", sft_checkpoint: "DPO 的初始 SFT 权重。",
  max_lr: "支持科学计数法，例如 3e-4。", max_learning_rate: "支持科学计数法，例如 2e-5。",
  min_lr: "支持科学计数法。", min_learning_rate: "支持科学计数法。",
  beta: "DPO 相对参考模型的偏离约束强度。", theta: "RoPE 的频率基数。"
};

const $ = (id) => document.getElementById(id);
const state = { configs: [], active: null, data: null, saved: "", view: "form", busy: false, nullable: new Set(), errors: new Map() };
let nextFieldId = 0;

function textElement(tag, className, content) {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = content;
  return element;
}

function pathKey(path) { return JSON.stringify(path); }
function json(value) { return JSON.stringify(value, null, 2); }
function isObject(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }

function setValue(path, value) {
  let target = state.data;
  for (const key of path.slice(0, -1)) target = target[key];
  target[path.at(-1)] = value;
}

function getValue(path) {
  return path.reduce((value, key) => value[key], state.data);
}

function isDirty() {
  return state.data !== null && (state.errors.size > 0 || json(state.data) !== state.saved);
}

function updateState() {
  const dirty = isDirty();
  $("save-state").textContent = state.busy ? "处理中…" : dirty ? "有未保存修改" : state.data ? "已保存" : "未读取";
  $("save-state").classList.toggle("dirty", dirty);
  $("save-button").disabled = state.busy || !dirty || state.errors.size > 0;
  $("reload-button").disabled = state.busy || !state.active;
  $("editor").inert = state.busy;
  $("field-search").disabled = state.busy;
  for (const button of document.querySelectorAll(".config-tab, .view-button")) button.disabled = state.busy;
}

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
  $("notice").hidden = !message;
}

async function request(url, options) {
  const response = await fetch(url, options);
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail || result));
  return result;
}

function collectNullable(value, path = []) {
  if (!isObject(value)) return;
  for (const [key, child] of Object.entries(value)) {
    const childPath = [...path, key];
    if (child === null) state.nullable.add(pathKey(childPath));
    else collectNullable(child, childPath);
  }
}

function renderTabs() {
  $("config-tabs").replaceChildren();
  for (const config of state.configs) {
    const button = textElement("button", "config-tab", config.label);
    button.type = "button";
    button.dataset.id = config.id;
    button.addEventListener("click", () => {
      if (config.id !== state.active?.id) loadConfig(config.id);
    });
    $("config-tabs").append(button);
  }
}

async function loadConfig(id, ask = true) {
  if (ask && isDirty() && !window.confirm("还有未保存的修改，放弃修改并读取配置？")) return;
  state.busy = true;
  updateState();
  notice("");
  try {
    const config = await request(`/api/configs/${encodeURIComponent(id)}`);
    state.active = config;
    state.data = config.data;
    state.saved = json(config.data);
    state.nullable.clear();
    state.errors.clear();
    collectNullable(state.data);
    $("config-title").textContent = config.label;
    $("config-path").textContent = config.filename;
    $("config-hint").hidden = !["sft", "dpo"].includes(config.id);
    $("field-search").value = "";
    for (const tab of document.querySelectorAll(".config-tab")) {
      const selected = tab.dataset.id === config.id;
      tab.classList.toggle("active", selected);
      tab.setAttribute("aria-current", selected ? "page" : "false");
    }
    renderForm();
    $("raw-json").value = json(state.data);
    $("raw-error").hidden = true;
    $("raw-json").classList.remove("invalid");
    $("empty-search").hidden = true;
  } catch (error) {
    notice(`读取失败：${error.message}`, true);
  } finally {
    state.busy = false;
    updateState();
  }
}

function setFieldError(path, input, errorElement, error) {
  const key = pathKey(path);
  if (error) state.errors.set(key, error);
  else state.errors.delete(key);
  input.classList.toggle("invalid", Boolean(error));
  errorElement.textContent = error || "";
  errorElement.hidden = !error;
  updateState();
}

function valueField(key, value, path) {
  const field = textElement("div", "parameter-field", "");
  field.dataset.search = `${labels[key] || key} ${path.join(".")}`.toLowerCase();
  const heading = textElement("div", "field-label", "");
  const inputId = `field-${++nextFieldId}`;
  const label = textElement("label", "", labels[key] || key);
  label.htmlFor = inputId;
  heading.append(label, textElement("span", "field-key", key));
  field.append(heading);
  const errorElement = textElement("p", "field-error", "");
  errorElement.hidden = true;

  if (Array.isArray(value)) {
    field.classList.add("wide");
    heading.append(textElement("span", "field-type", "JSON 数组"));
    const input = textElement("textarea", "json-input", "");
    input.id = inputId;
    input.value = json(value);
    input.spellcheck = false;
    input.addEventListener("input", () => {
      try {
        const parsed = JSON.parse(input.value);
        if (!Array.isArray(parsed)) throw new Error("这里需要 JSON 数组，例如 [\"train\"]。");
        setValue(path, parsed);
        setFieldError(path, input, errorElement, null);
      } catch (error) { setFieldError(path, input, errorElement, error.message); }
    });
    field.append(input);
  } else if (typeof value === "boolean") {
    const wrapper = textElement("label", "boolean-input", "");
    wrapper.htmlFor = inputId;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = inputId;
    input.checked = value;
    const status = textElement("span", "boolean-label", value ? "开启 · true" : "关闭 · false");
    input.addEventListener("change", () => {
      setValue(path, input.checked);
      status.textContent = input.checked ? "开启 · true" : "关闭 · false";
      updateState();
    });
    wrapper.append(input, status);
    field.append(wrapper);
  } else {
    const nullable = state.nullable.has(pathKey(path));
    let kind = typeof value === "number" ? "number" : "string";
    const multiline = typeof value === "string" && value.includes("\n");
    if (multiline) field.classList.add("wide");
    const input = document.createElement(multiline ? "textarea" : "input");
    input.className = "value-input";
    input.id = inputId;
    input.value = value === null ? "" : String(value);
    if (multiline) input.rows = 4;
    else input.type = "text";
    if (kind === "number") input.inputMode = "decimal";
    if (nullable) input.placeholder = "未设置；输入值启用，清空为 null";
    const parseInput = () => {
      let nextValue = input.value;
      let error = null;
      if (nullable && nextValue === "") nextValue = null;
      else if (kind === "number") {
        if (nextValue.trim() === "" || !Number.isFinite(Number(nextValue))) error = "请输入有效数字，支持 3e-4 这样的科学计数法。";
        else nextValue = Number(nextValue);
      }
      if (!error) setValue(path, nextValue);
      setFieldError(path, input, errorElement, error);
    };
    input.addEventListener("input", parseInput);
    if (nullable) {
      const wrapper = textElement("div", "nullable-input", "");
      const select = document.createElement("select");
      select.setAttribute("aria-label", `${labels[key] || key}的非空值类型`);
      for (const [optionValue, optionLabel] of [["string", "文本"], ["number", "数字"]]) {
        const option = textElement("option", "", optionLabel);
        option.value = optionValue;
        select.append(option);
      }
      select.value = kind;
      select.addEventListener("change", () => { kind = select.value; input.inputMode = kind === "number" ? "decimal" : "text"; parseInput(); });
      wrapper.append(select, input);
      field.append(wrapper);
    } else {
      heading.append(textElement("span", "field-type", kind === "number" ? "数字" : "文本"));
      field.append(input);
    }
  }
  if (hints[key]) field.append(textElement("p", "field-hint", hints[key]));
  field.append(errorElement);
  return field;
}

function objectFields(value, path) {
  const grid = textElement("div", "field-grid", "");
  for (const [key, child] of Object.entries(value)) {
    const childPath = [...path, key];
    if (isObject(child)) {
      const nested = textElement("details", "nested-object", "");
      nested.open = true;
      const summary = textElement("summary", "", labels[key] || key);
      summary.append(textElement("span", "field-key", key));
      nested.append(summary, objectFields(child, childPath));
      grid.append(nested);
    } else grid.append(valueField(key, child, childPath));
  }
  return grid;
}

function sectionJson(path) {
  const input = textElement("textarea", "json-input", "");
  input.value = json(getValue(path));
  input.spellcheck = false;
  input.setAttribute("aria-label", `${path.join(".")} JSON`);
  const errorElement = textElement("p", "field-error", "");
  errorElement.hidden = true;
  input.addEventListener("input", () => {
    try {
      const value = JSON.parse(input.value);
      if (!isObject(value)) throw new Error("这一组参数需要 JSON 对象。");
      setValue(path, value);
      collectNullable(value, path);
      setFieldError(path, input, errorElement, null);
    } catch (error) { setFieldError(path, input, errorElement, error.message); }
  });
  const wrapper = textElement("div", "section-json", "");
  wrapper.append(input, errorElement);
  return wrapper;
}

function renderForm() {
  const form = $("form-editor");
  form.replaceChildren();
  nextFieldId = 0;
  for (const [key, value] of Object.entries(state.data)) {
    const section = textElement("section", "parameter-section", "");
    section.dataset.search = `${labels[key] || key} ${key}`.toLowerCase();
    const heading = textElement("div", "section-heading", "");
    heading.append(textElement("h3", "", labels[key] || key), textElement("code", "", key));
    section.append(heading);
    if (isObject(value)) {
      const toggle = textElement("button", "section-toggle", "编辑 JSON");
      toggle.type = "button";
      let asJson = false;
      let content = objectFields(value, [key]);
      toggle.addEventListener("click", () => {
        if (state.errors.size) { notice("先修正标红的输入，再切换编辑方式。", true); return; }
        asJson = !asJson;
        const replacement = asJson ? sectionJson([key]) : objectFields(getValue([key]), [key]);
        content.replaceWith(replacement);
        content = replacement;
        toggle.textContent = asJson ? "参数表单" : "编辑 JSON";
        applySearch();
      });
      heading.append(toggle);
      section.append(content);
    } else section.append(valueField(key, value, [key]));
    form.append(section);
  }
  applySearch();
}

function applySearch() {
  const query = $("field-search").value.trim().toLowerCase();
  let visibleSections = 0;
  for (const section of document.querySelectorAll(".parameter-section")) {
    const matchSection = section.dataset.search.includes(query);
    for (const field of section.querySelectorAll(".parameter-field")) field.hidden = !matchSection && !field.dataset.search.includes(query);
    for (const nested of Array.from(section.querySelectorAll(".nested-object")).reverse()) nested.hidden = !Array.from(nested.querySelectorAll(".parameter-field")).some((field) => !field.hidden);
    const fields = [...section.querySelectorAll(".parameter-field")];
    section.hidden = fields.length > 0 ? !fields.some((field) => !field.hidden) : Boolean(query) && !matchSection;
    if (!section.hidden) visibleSections += 1;
  }
  $("empty-search").hidden = state.view !== "form" || visibleSections > 0 || !query;
}

function switchView(view) {
  if (view === state.view || !state.data) return;
  if (state.errors.size) { notice("先修正标红的输入，再切换编辑方式。", true); return; }
  state.view = view;
  const form = view === "form";
  $("form-editor").hidden = !form;
  $("json-editor").hidden = form;
  $("search-control").hidden = !form;
  $("form-view-button").classList.toggle("active", form);
  $("json-view-button").classList.toggle("active", !form);
  $("form-view-button").setAttribute("aria-pressed", String(form));
  $("json-view-button").setAttribute("aria-pressed", String(!form));
  if (form) { collectNullable(state.data); renderForm(); }
  else { $("raw-json").value = json(state.data); $("empty-search").hidden = true; }
}

async function saveConfig() {
  if (!isDirty() || state.errors.size) return;
  state.busy = true;
  updateState();
  notice("");
  try {
    const result = await request(`/api/configs/${encodeURIComponent(state.active.id)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ data: state.data })
    });
    state.data = result.data;
    state.saved = json(result.data);
    $("raw-json").value = state.saved;
    notice(`已保存到 ${result.filename}`);
  } catch (error) { notice(`保存失败：${error.message}`, true); }
  finally { state.busy = false; updateState(); }
}

$("form-editor").addEventListener("submit", (event) => { event.preventDefault(); saveConfig(); });
$("field-search").addEventListener("input", applySearch);
$("form-view-button").addEventListener("click", () => switchView("form"));
$("json-view-button").addEventListener("click", () => switchView("json"));
$("save-button").addEventListener("click", saveConfig);
$("reload-button").addEventListener("click", () => loadConfig(state.active.id));
$("raw-json").addEventListener("input", () => {
  try {
    const data = JSON.parse($("raw-json").value);
    if (!isObject(data)) throw new Error("配置根节点需要是 JSON 对象。");
    state.data = data;
    setFieldError(["$raw"], $("raw-json"), $("raw-error"), null);
  } catch (error) { setFieldError(["$raw"], $("raw-json"), $("raw-error"), error.message); }
});

async function initialize() {
  state.busy = true;
  updateState();
  try {
    const result = await request("/api/configs");
    const order = ["pretrain", "sft", "dpo", "data_pipeline", "label"];
    state.configs = result.configs.sort((a, b) => {
      const rank = (id) => order.includes(id) ? order.indexOf(id) : order.length;
      return rank(a.id) - rank(b.id);
    });
    renderTabs();
    if (state.configs.length) await loadConfig(state.configs[0].id, false);
    else notice("没有可编辑的配置文件。", true);
  } catch (error) { notice(`连接失败：${error.message}。请通过配置面板服务打开此页面。`, true); }
  finally { state.busy = false; updateState(); }
}

initialize();
