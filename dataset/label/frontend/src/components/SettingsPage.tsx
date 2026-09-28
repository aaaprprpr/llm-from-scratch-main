import { useEffect, useState } from "react";
import { requestJson } from "../api";

type Source = "deepseek" | "qwen_api" | "local" | "deepseek_web" | "qwen_web";
type Choice = { single: Source; batch: Source[]; failure_fallback: boolean };
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
  const [saving, setSaving] = useState<"single" | "batch" | "fallback" | null>(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const value = await requestJson<Settings>("/api/settings/models");
        if (!active) return;
        if (typeof value.failure_fallback !== "boolean") {
          setMessage("当前清洗台后端仍是旧版本。请暂停批量任务后重启 ./clean。");
          return;
        }
        setSettings(value);
        setDraft((current) => current ?? { single: value.single, batch: value.batch,
          failure_fallback: value.failure_fallback });
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

  async function save(kind: "single" | "batch" | "fallback") {
    if (!draft || !settings) return;
    setSaving(kind);
    setMessage("");
    try {
      const selection = {
        single: kind === "single" ? draft.single : settings.single,
        batch: kind === "batch" ? draft.batch : settings.batch,
        failure_fallback: kind === "fallback" ? draft.failure_fallback : settings.failure_fallback,
      };
      const value = await requestJson<Settings>("/api/settings/models", {
        method: "PUT", body: JSON.stringify(selection),
      });
      setSettings(value);
      setDraft((current) => current && (kind === "single"
        ? { ...current, single: value.single }
        : kind === "batch" ? { ...current, batch: value.batch }
          : { ...current, failure_fallback: value.failure_fallback }));
      setMessage(kind === "single" ? "单条清洗模型已保存。"
        : kind === "batch" ? "批量清洗模型已保存，后续任务按新设置执行。"
          : "失败补救设置已保存，下次启动或续跑批量任务时生效。");
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
        <div className="settings-card">
          <label className="settings-batch-option">
            <input type="checkbox" checked={draft.failure_fallback} disabled={saving !== null}
              onChange={(event) => setDraft({ ...draft, failure_fallback: event.target.checked })} />
            <span>普通清洗失败后，交给 DeepSeek API 补一次</span>
          </label>
          <p className="muted">仅用于批量清洗：勾选的模型仍未完成时补救。曾触发内容敏感拒绝的条目不会发送；正常新条目不会交给这个备用来源。{!settings.available.deepseek?.configured && "当前未配置 DeepSeek API Key，开启后也不会调用。"}</p>
          <button className="primary" onClick={() => void save("fallback")}
            disabled={saving !== null || draft.failure_fallback === settings.failure_fallback}>
            {saving === "fallback" ? "保存中…" : "保存失败补救设置"}
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
