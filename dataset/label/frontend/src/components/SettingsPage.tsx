import { useEffect, useState } from "react";
import { requestJson } from "../api";

type Source = "deepseek" | "qwen_api" | "local";
type Choice = { single: Source; batch: Source };
type Settings = Choice & {
  available: Record<Source, { model: string; configured: boolean }>;
  local_model: { state: string; detail: string };
};

const choices: { value: Source; label: string }[] = [
  { value: "deepseek", label: "DeepSeek API" },
  { value: "qwen_api", label: "Qwen API" },
  { value: "local", label: "本地 Qwen 27B" },
];

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState<Choice | null>(null);
  const [saving, setSaving] = useState(false);
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

  async function save() {
    if (!draft) return;
    setSaving(true);
    setMessage("");
    try {
      const value = await requestJson<Settings>("/api/settings/models", {
        method: "PUT", body: JSON.stringify(draft),
      });
      setSettings(value);
      setDraft({ single: value.single, batch: value.batch });
      setMessage("模型来源已保存，后续清洗按新设置执行。已有清洗进度保留。");
    } catch (error) {
      setMessage(`保存失败：${String(error)}`);
    } finally {
      setSaving(false);
    }
  }

  return <main className="document-panel panel">
    <section className="settings-page">
      <h2>设置</h2>
      <p className="muted">分别选择手动触发的单条清洗和自动批量清洗所用的模型。已有审核和批量进度保持原样。DeepSeek 遇到明确的内容敏感拒绝时，仍按现有流程尝试一次 Qwen API。</p>
      {settings && draft ? <>
        <div className="settings-card">
          <label>单条清洗模型
            <select value={draft.single} onChange={(event) => setDraft({ ...draft, single: event.target.value as Source })}>
              {choices.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </label>
          <label>批量清洗模型
            <select value={draft.batch} onChange={(event) => setDraft({ ...draft, batch: event.target.value as Source })}>
              {choices.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </label>
          <button className="primary" onClick={() => void save()} disabled={saving ||
            (draft.single === settings.single && draft.batch === settings.batch)}>
            {saving ? "保存中…" : "保存设置"}
          </button>
        </div>
        <div className="settings-card settings-sources">
          <h3>模型状态</h3>
          {choices.map((item) => <div key={item.value} className="settings-source-row">
            <strong>{item.label}</strong>
            <span>{settings.available[item.value].model}</span>
            <span>{item.value === "local" ? settings.local_model.detail :
              settings.available[item.value].configured ? "API Key 已配置" : "未配置 API Key"}</span>
          </div>)}
          <p className="muted">本地服务在打开清洗台时自动检测并启动；模型首次加载需要稍等。切换批量模型需先暂停运行中的任务。</p>
        </div>
      </> : <p className="muted">正在读取设置…</p>}
      {message && <p className={message.startsWith("保存失败") || message.startsWith("读取设置失败") || message.startsWith("当前清洗台") ? "settings-message error" : "settings-message"}>{message}</p>}
    </section>
  </main>;
}
