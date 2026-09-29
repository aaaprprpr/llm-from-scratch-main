import { existsSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = fileURLToPath(new URL('.', import.meta.url));
function envValue(name) {
  if (process.env[name]) return process.env[name];
  try {
    const line = readFileSync(new URL('.env', import.meta.url), 'utf8').split(/\r?\n/)
      .find((item) => new RegExp(`^\\s*${name}\\s*=`).test(item));
    if (!line) return undefined;
    const value = line.slice(line.indexOf('=') + 1).trim();
    return value.startsWith('"') ? JSON.parse(value) : value.replace(/^'|'$/g, '');
  } catch (error) { if (error?.code === 'ENOENT') return undefined; throw error; }
}
const authPath = resolve(dir, envValue('KIMI_WEB_AUTH_FILE') || '.kimi-web-auth.json');
const statePath = resolve(dir, '.kimi-session.json');
const flags = new Set(['--new', '--continue', '--delete-after', '--delete-current', '--json-stdin', '--delete-session-stdin']);
const args = process.argv.slice(2);
const prompt = args.filter((arg) => !flags.has(arg)).join(' ').trim();
const fresh = args.includes('--new');
const continuing = args.includes('--continue');
const deleteAfter = args.includes('--delete-after');
const deleteCurrent = args.includes('--delete-current');
const jsonStdin = args.includes('--json-stdin');
const deleteSessionStdin = args.includes('--delete-session-stdin');
if ((!deleteCurrent && !jsonStdin && !deleteSessionStdin && !prompt) || (fresh && continuing) ||
    ((jsonStdin || deleteSessionStdin) && (prompt || fresh || continuing || deleteCurrent || deleteAfter)) ||
    (deleteCurrent && (prompt || fresh || continuing || deleteAfter))) {
  console.error('用法：node kimi_chat.mjs [--new | --continue | --delete-after] "你好"；或 node kimi_chat.mjs --delete-current');
  process.exit(1);
}
function readAuth() {
  let auth;
  try { auth = JSON.parse(readFileSync(authPath, 'utf8')); }
  catch (error) { if (error?.code === 'ENOENT') throw new Error('缺少 Kimi 凭据；先运行 python kimi_capture_auth.py <调试端口>'); throw error; }
  if (!auth.accessToken || !auth.refreshToken) throw new Error('Kimi 凭据文件无效，请重新导出');
  return auth;
}
function readState() {
  try {
    const state = JSON.parse(readFileSync(statePath, 'utf8'));
    if (!/^[a-f0-9-]{36}$/.test(state.chatId)) throw new Error('Kimi 会话状态文件无效');
    return state;
  } catch (error) { if (error?.code === 'ENOENT') throw new Error('没有可继续或删除的 Kimi 会话'); throw error; }
}
function saveState(chatId) { writeFileSync(statePath, `${JSON.stringify({ chatId }, null, 2)}\n`, { mode: 0o600 }); }
function clearState(chatId) {
  if (existsSync(statePath) && readState().chatId === chatId) unlinkSync(statePath);
}
function tokenExpiresSoon(token) {
  try { return JSON.parse(Buffer.from(token.split('.')[1], 'base64url')).exp * 1000 < Date.now() + 60_000; }
  catch { return false; }
}
function headers(auth, contentType) {
  return {
    authorization: `Bearer ${auth.accessToken}`,
    'content-type': contentType,
    ...(contentType === 'application/connect+json' ? { 'connect-protocol-version': '1' } : {}),
    'x-traffic-id': randomUUID().replaceAll('-', '').slice(0, 20),
  };
}
async function refresh(auth) {
  const { authorization, ...base } = headers(auth, 'application/json');
  const response = await fetch('https://auth.kimi.com/api/account.gateway.v1.AuthService/RefreshToken', {
    method: 'POST', headers: base, body: JSON.stringify({ refresh_token: auth.refreshToken }),
  });
  const value = await response.json();
  if (!response.ok || !value.accessToken || !value.refreshToken) throw new Error(`Kimi 登录刷新失败：HTTP ${response.status}；请重新导出凭据`);
  auth.accessToken = value.accessToken;
  auth.refreshToken = value.refreshToken;
  writeFileSync(authPath, `${JSON.stringify(auth, null, 2)}\n`, { mode: 0o600 });
}
async function request(auth, url, body, contentType) {
  if (tokenExpiresSoon(auth.accessToken)) await refresh(auth);
  let response = await fetch(url, { method: 'POST', headers: headers(auth, contentType), body });
  if (response.status === 401) {
    await refresh(auth);
    response = await fetch(url, { method: 'POST', headers: headers(auth, contentType), body });
  }
  return response;
}
async function deleteChat(auth, chatId) {
  const response = await request(auth, 'https://www.kimi.com/apiv2/kimi.chat.v1.ChatService/DeleteChat', JSON.stringify({ chat_id: chatId }), 'application/json');
  const value = await response.json();
  if (!response.ok || value.chatId !== chatId) throw new Error(`Kimi 删除失败：HTTP ${response.status}`);
}
function framedJson(value) {
  const json = Buffer.from(JSON.stringify(value));
  const frame = Buffer.allocUnsafe(json.length + 5);
  frame[0] = 0;
  frame.writeUInt32BE(json.length, 1);
  json.copy(frame, 5);
  return frame;
}
async function chat(auth, question, chatId, structured = false, onCreated = () => {}) {
  const payload = {
    scenario: 'SCENARIO_CHAT',
    tools: [],
    message: { role: 'user', blocks: [{ message_id: '', text: { content: question } }], scenario: 'SCENARIO_CHAT', is_goal: false },
    options: { thinking: false, enable_plugin: false, reasoning_effort: 'REASONING_EFFORT_NONE', model: 'k2d6-chat' },
    project_id: '',
  };
  if (chatId) payload.chat_id = chatId;
  const response = await request(auth, 'https://www.kimi.com/apiv2/kimi.gateway.chat.v1.ChatService/Chat', framedJson(payload), 'application/connect+json');
  if (!response.ok || !response.headers.get('content-type')?.includes('application/connect+json')) {
    throw new Error(`Kimi 聊天失败：HTTP ${response.status}, ${(await response.text()).slice(0, 200)}`);
  }
  let pending = Buffer.alloc(0);
  let answer = '';
  const blocks = new Map();
  let actualId = chatId;
  let errorMessage;
  for await (const part of response.body) {
    pending = Buffer.concat([pending, Buffer.from(part)]);
    while (pending.length >= 5) {
      const length = pending.readUInt32BE(1);
      if (length > 16_000_000) throw new Error('Kimi 返回的帧过大');
      if (pending.length < length + 5) break;
      const flags = pending[0];
      const value = JSON.parse(pending.subarray(5, length + 5).toString('utf8'));
      pending = pending.subarray(length + 5);
      if (value.chat?.id) {
        if (!actualId) onCreated(value.chat.id);
        actualId = value.chat.id;
      }
      if (value.error || (flags & 2 && value.code && value.code !== 'ok')) errorMessage = value.error?.message ?? value.message ?? `Connect 错误：${JSON.stringify(value).slice(0, 200)}`;
      const text = value.block?.text?.content;
      if (typeof text === 'string' && (value.op === 'append' || value.op === 'set')) {
        const id = value.block?.id ?? 'main';
        const prior = blocks.get(id) ?? '';
        const delta = value.op === 'append' ? text : text.startsWith(prior) ? text.slice(prior.length) : text;
        blocks.set(id, value.op === 'append' ? prior + text : text);
        if (delta) { if (!structured) process.stdout.write(delta); answer += delta; }
      }
    }
  }
  if (answer && !structured) process.stdout.write('\n');
  if (errorMessage) throw new Error(errorMessage);
  if (!actualId || !answer) throw new Error('Kimi 响应缺少会话 ID 或正文');
  return { chatId: actualId, content: answer };
}

let createdId;
try {
  const auth = readAuth();
  if (deleteSessionStdin) {
    const { session_id: chatId } = JSON.parse(readFileSync(0, 'utf8'));
    if (typeof chatId !== 'string' || !/^[a-f0-9-]{36}$/.test(chatId)) throw new Error('Kimi 会话 ID 无效');
    await deleteChat(auth, chatId);
    console.log(JSON.stringify({ deleted: chatId }));
  } else if (deleteCurrent) {
    const { chatId } = readState();
    await deleteChat(auth, chatId);
    clearState(chatId);
    console.error(`已删除 Kimi 会话 ${chatId}`);
  } else {
    let question = prompt;
    let prior;
    if (jsonStdin) {
      const request = JSON.parse(readFileSync(0, 'utf8'));
      if (typeof request.prompt !== 'string' || !request.prompt.trim()) throw new Error('缺少提示词');
      if (request.session_id != null && (typeof request.session_id !== 'string' || !/^[a-f0-9-]{36}$/.test(request.session_id)))
        throw new Error('Kimi 会话 ID 无效');
      question = request.prompt;
      prior = request.session_id || undefined;
    } else {
      prior = continuing || (!fresh && !deleteAfter && existsSync(statePath)) ? readState().chatId : undefined;
    }
    const result = await chat(auth, question, prior, jsonStdin, (id) => {
      if (jsonStdin) console.log(JSON.stringify({ session_id: id }));
    });
    createdId = result.chatId;
    if (jsonStdin) console.log(JSON.stringify({ session_id: result.chatId, content: result.content }));
    else {
      console.error(`会话 ID: ${result.chatId}`);
      if (!deleteAfter) saveState(result.chatId);
    }
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
} finally {
  if (deleteAfter && createdId) {
    try {
      await deleteChat(readAuth(), createdId);
      clearState(createdId);
      console.error(`已删除本次 Kimi 会话 ${createdId}`);
    } catch (error) { console.error(`删除 Kimi 会话 ${createdId} 失败：${error.message}`); process.exitCode = 1; }
  }
}
