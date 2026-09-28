import { useEffect, useState } from "react";
import { requestJson } from "../api";

type Source = "deepseek" | "qwen_api" | "local" | "deepseek_web" | "qwen_web";
type Choice = { single: Source; batch: Source[] };
type Settings = Choice & {
  available: Record<Source, { model: string; configured: boolean }>;
  local_model: { state: string; detail: string };
};

const choices: { value: Source; label: string }[] = [
  { value: "deepseek", label: "DeepSeek API" },
  { value: "deepseek_web", label: "DeepSeek 网页" },
  { value: "qwen_web", label: "千问网页" },
  { value: "qwen_api", label: "Qwen API" },
  { value: "local", label: "本地 Qwen 27B" },
];

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState<Choice | null>(null);
  const [saving, setSaving] = useState<"single" | "batch" | null>(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const value = await requestJson<Settings>("/api/settings/models");
        if (!active) return;
        setSettings(value);
        setDraft((current) => current ?? { single: value.single, batch: value.batch });
      } catch (error) {
        if (active) setMessage(String(error).includes("404")
          ? "当前清洗台后端仍是旧版本。请停止旧服务，再运行 ./clean。"
          : `读取设置失败：${String(error)}`);
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  function toggleBatch(source: Source) {
    setDraft((current) => current && ({ ...current, batch: current.batch.includes(source)
      ? current.batch.filter((item) => item !== source)
      : [...current.batch, source] }));
  }

  async function save(kind: "single" | "batch") {
    if (!draft || !settings) return;
    setSaving(kind);
    setMessage("");
    try {
      const selection = kind === "single"
        ? { single: draft.single, batch: settings.batch }
        : { single: settings.single, batch: draft.batch };
      const value = await requestJson<Settings>("/api/settings/models", {
        method: "PUT", body: JSON.stringify(selection),
      });
      setSettings(value);
      setDraft((current) => current && (kind === "single"
        ? { ...current, single: value.single }
        : { ...current, batch: value.batch }));
      setMessage(kind === "single" ? "单条清洗模型已保存。" : "批量清洗模型已保存，后续任务按新设置执行。");
    } catch (error) {
      setMessage(`保存失败：${String(error)}`);
    } finally {
      setSaving(null);
    }
  }

  return <main className="document-panel panel">
    <section className="settings-page">
      <h2>设置</h2>
      <p className="muted">单条清洗选一个模型；批量清洗勾选要混合使用的来源。已有审核和批量进度保持原样。DeepSeek API 遇到明确的内容敏感拒绝时，仍按现有流程尝试一次 Qwen API。</p>
      {settings && draft ? <>
        <div className="settings-card">
          <label>单条清洗模型
            <select value={draft.single} disabled={saving !== null}
              onChange={(event) => setDraft({ ...draft, single: event.target.value as Source })}>
              {choices.map((item) => <option key={item.value} value={item.value} disabled={!settings.available[item.value]}>{item.label}</option>)}
            </select>
          </label>
          <button className="primary" onClick={() => void save("single")}
            disabled={saving !== null || draft.single === settings.single}>
            {saving === "single" ? "保存中…" : "保存单条模型"}
          </button>
        </div>
        <div className="settings-card">
          <fieldset className="settings-batch-options" disabled={saving !== null}>
            <legend>批量清洗模型（可多选）</legend>
            {choices.map((item) => <label key={item.value} className="settings-batch-option">
              <input type="checkbox" checked={draft.batch.includes(item.value)}
                disabled={!settings.available[item.value]?.configured && !draft.batch.includes(item.value)}
                onChange={() => toggleBatch(item.value)} />
              <span>{item.label}</span>
            </label>)}
          </fieldset>
          <button className="primary" onClick={() => void save("batch")}
            disabled={saving !== null || !draft.batch.length || draft.batch.join(",") === settings.batch.join(",")}>
            {saving === "batch" ? "保存中…" : "保存批量模型"}
          </button>
        </div>
        <div className="settings-card settings-sources">
          <h3>模型状态</h3>
          {choices.map((item) => <div key={item.value} className="settings-source-row">
            <strong>{item.label}</strong>
            <span>{settings.available[item.value]?.model ?? "重启清洗台后可用"}</span>
            <span>{item.value === "local" ? settings.local_model.detail :
              item.value === "deepseek_web"
                ? settings.available[item.value]?.configured ? "网页登录令牌已配置" : "未配置 DEEPSEEK_WEB_TOKEN"
                : item.value === "qwen_web"
                  ? settings.available[item.value]?.configured ? "网页凭据已配置" : "未找到 QWEN_WEB_AUTH_FILE"
                  : settings.available[item.value]?.configured ? "API Key 已配置" : "未配置 API Key"}</span>
          </div>)}
          <p className="muted">本地服务在打开清洗台时自动检测并启动；模型首次加载需要稍等。切换批量模型需先暂停运行中的任务。</p>
        </div>
      </> : <p className="muted">正在读取设置…</p>}
      {message && <p className={message.startsWith("保存失败") || message.startsWith("读取设置失败") || message.startsWith("当前清洗台") ? "settings-message error" : "settings-message"}>{message}</p>}
    </section>
  </main>;
}
