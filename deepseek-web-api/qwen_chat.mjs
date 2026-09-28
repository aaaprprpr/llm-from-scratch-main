import { readFileSync, writeFileSync, unlinkSync, existsSync, openSync, closeSync } from 'node:fs';
import { createHmac, randomUUID } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

function envValue(name) {
  if (process.env[name]) return process.env[name];
  try {
    const line = readFileSync(new URL('.env', import.meta.url), 'utf8').split(/\r?\n/)
      .find((item) => new RegExp(`^\\s*${name}\\s*=`).test(item));
    if (!line) return undefined;
    const value = line.slice(line.indexOf('=') + 1).trim();
    if (value.startsWith('"')) return JSON.parse(value);
    if (value.startsWith("'") && value.endsWith("'")) return value.slice(1, -1);
    return value;
  } catch (error) {
    if (error?.code === 'ENOENT') return undefined;
    throw error;
  }
}

const AUTH_PATH = resolve(fileURLToPath(new URL('.', import.meta.url)), envValue('QWEN_WEB_AUTH_FILE') || '.qwen-web-auth.json');
const STATE_URL = new URL('.qwen-session.json', import.meta.url);
const args = process.argv.slice(2);
const continueChat = args.includes('--continue');
const newChat = args.includes('--new');
const deleteCurrent = args.includes('--delete-current');
const deleteAfter = args.includes('--delete-after');
const jsonStdin = args.includes('--json-stdin');
const deleteSessionStdin = args.includes('--delete-session-stdin');
const prompt = args.filter((arg) => !['--continue', '--new', '--delete-current', '--delete-after', '--json-stdin', '--delete-session-stdin'].includes(arg)).join(' ').trim();

if ((!deleteCurrent && !jsonStdin && !deleteSessionStdin && !prompt) || (newChat && continueChat) ||
    ((jsonStdin || deleteSessionStdin) && (prompt || continueChat || newChat || deleteCurrent || deleteAfter)) ||
    (deleteCurrent && (prompt || continueChat || newChat || deleteAfter))) {
  console.error('用法：node qwen_chat.mjs [--new | --continue | --delete-after] "你好"；或 node qwen_chat.mjs --delete-current');
  process.exit(1);
}

function readAuth(requireSignature = true) {
  let auth;
  try { auth = JSON.parse(readFileSync(AUTH_PATH, 'utf8')); }
  catch (error) {
    if (error?.code === 'ENOENT') throw new Error('缺少千问凭据；先运行 python qwen_capture_auth.py <调试端口>');
    throw error;
  }
  const scene = auth.storage?.scene_list?.qwen_chat;
  if (!auth.query || !auth.cookies?.tongyi_sso_ticket_hash || requireSignature && !scene?.['eo-clt-actkn']) {
    throw new Error('千问凭据文件无效，请重新导出');
  }
  return auth;
}

function saveAuth(auth) {
  writeFileSync(AUTH_PATH, `${JSON.stringify(auth, null, 2)}\n`, { mode: 0o600 });
}

function readState() {
  try {
    const state = JSON.parse(readFileSync(STATE_URL, 'utf8'));
    if (!/^[a-f0-9]{32}$/.test(state.sessionId) || typeof state.reqId !== 'string') throw new Error('会话状态文件无效');
    return state;
  } catch (error) {
    if (error?.code === 'ENOENT') throw new Error('没有可继续或删除的会话；先发送一条消息');
    throw error;
  }
}

function saveState(state) {
  writeFileSync(STATE_URL, `${JSON.stringify(state, null, 2)}\n`, { mode: 0o600 });
}

function clearState() {
  try { unlinkSync(STATE_URL); } catch (error) { if (error?.code !== 'ENOENT') throw error; }
}

function commonUrl(auth, host, path) {
  const url = new URL(`https://${host}${path}`);
  for (const [key, value] of new URLSearchParams(auth.query)) url.searchParams.set(key, value);
  return url;
}

function cookieHeaders(auth) {
  return {
    cookie: Object.entries(auth.cookies).map(([key, value]) => `${key}=${value}`).join('; '),
    'content-type': 'application/json',
    'x-xsrf-token': decodeURIComponent(auth.cookies['XSRF-TOKEN'] ?? ''),
    origin: 'https://www.qianwen.com',
    referer: 'https://www.qianwen.com/',
  };
}

async function refreshSignaturePool(auth) {
  const scenes = auth.storage.scene_list;
  const chat = scenes.qwen_chat;
  const remaining = chat['eo-clt-bacsft']?.length ?? 0;
  const expiresInMs = (chat['eo-clt-actkn-dl'] ?? 0) * 1000 - Date.now();
  if (remaining > 20 && expiresInMs > 30 * 60 * 1000) return;

  const base = scenes.qwen_web;
  const query = new URLSearchParams({
    businessScene: 'qwen_web',
    unifyRelateGenerate: 'voice_command,qwen_chat',
    chid: randomUUID().replaceAll('-', ''),
  });
  const url = `https://sec.qianwen.com/security/external/access/refresh?${query}`;
  try {
    const response = await fetch(url, {
      headers: {
        ...cookieHeaders(auth),
        'eo-clt-dvidn': base['eo-clt-dvidn'],
        'eo-clt-actkn': base['eo-clt-actkn'],
        'eo-clt-sftcnt': '100',
        'clt-acs-caer': 'vrad',
        'eo-clt-acs-bx-intss': '2',
      },
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== 0 || !payload.data?.['eo-clt-bacsft']?.length) {
      throw new Error(`HTTP ${response.status}, status=${payload.status}, msg=${payload.msg ?? ''}`);
    }
    const { unifyRelate, ...baseData } = payload.data;
    const refreshedChat = unifyRelate?.find((item) => item.businessScene === 'qwen_chat');
    if (!refreshedChat?.['eo-clt-bacsft']?.length) throw new Error('补充结果缺少 qwen_chat 签名材料');
    scenes.qwen_web = { ...base, ...baseData };
    for (const item of unifyRelate ?? []) {
      if (item.businessScene) scenes[item.businessScene] = { ...scenes[item.businessScene], ...item };
    }
    saveAuth(auth);
    console.error(`千问签名材料已自动补充（${scenes.qwen_chat['eo-clt-bacsft'].length} 项）`);
  } catch (error) {
    if (remaining > 0 && expiresInMs > 0) {
      console.error(`签名材料自动补充失败，先使用剩余 ${remaining} 项：${error instanceof Error ? error.message : String(error)}`);
      return;
    }
    throw new Error(`千问签名材料已用完或过期，自动补充失败：${error instanceof Error ? error.message : String(error)}。请检查网页登录状态，必要时重新运行 qwen_capture_auth.py`);
  }
}

async function deleteSession(auth, sessionId) {
  const url = commonUrl(auth, 'chat2-api.qianwen.com', '/api/v1/session/delete');
  const response = await fetch(url, {
    method: 'POST', headers: cookieHeaders(auth),
    body: JSON.stringify({ session_id: sessionId, biz_id: 'ai_qwen' }),
  });
  const payload = await response.json();
  if (!response.ok || payload.code !== 0 || payload.success !== true) {
    throw new Error(`删除会话失败：HTTP ${response.status}, code=${payload.code}, msg=${payload.msg ?? ''}`);
  }
}

async function signedRequest(auth, state, prompt) {
  // Several batch slots consume signatures from one credential file. Reserve
  // one under a short file lock, then release it before the network request.
  let lock;
  for (let attempt = 0; attempt < 600; attempt++) {
    try { lock = openSync(`${AUTH_PATH}.lock`, 'wx', 0o600); break; }
    catch (error) {
      if (error?.code !== 'EEXIST') throw error;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  if (lock === undefined) throw new Error('千问凭据文件被占用');
  let sac;
  try {
    const current = readAuth(false);
    await refreshSignaturePool(current);
    sac = current.storage.scene_list.qwen_chat['eo-clt-bacsft']?.pop();
    if (!sac) throw new Error('千问签名材料已用完，自动补充未成功');
    saveAuth(current);
    Object.assign(auth, current);
  } finally {
    closeSync(lock);
    unlinkSync(`${AUTH_PATH}.lock`);
  }
  const scene = auth.storage.scene_list.qwen_chat;
  const base = auth.storage.scene_list.qwen_web;

  const reqId = randomUUID().replaceAll('-', '');
  const body = JSON.stringify({
    req_id: reqId, parent_req_id: state.reqId,
    messages: [{ mime_type: 'text/plain', content: prompt, meta_data: { ori_query: prompt }, status: 'complete' }],
    scene: 'chat', sub_scene: '', scene_param: state.reqId === '0' ? 'first_turn' : 'continue_chat',
    session_id: state.sessionId, biz_id: 'ai_qwen', topic_id: state.topicId,
    model: 'Qwen', from: 'default', protocol_version: 'v2', messages_merge: false,
    chat_client: 'h5', deep_search: null, temporary: false,
    params_extra: { supports_cowork: 'true' }, chat_mode: 'quick', bucket: {},
  });
  const url = new URL('https://chat2.qianwen.com/api/v2/chat');
  url.searchParams.set('biz_id', 'ai_qwen');
  url.searchParams.set('fe_version', '1.0.0');
  for (const [key, value] of new URLSearchParams(auth.query)) url.searchParams.set(key, value);
  const signData = Object.fromEntries(url.searchParams);
  const timestamp = Date.now();
  const nonce = Array.from({ length: 11 }, () => '0123456789abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 36)]).join('');
  signData.nonce = nonce;
  signData.timestamp = timestamp;
  const secret = `${sac}:${timestamp}`;
  const hmac = (value) => createHmac('sha256', secret).update(value).digest('base64');
  const bodySign = hmac(body);
  const version = '1.0.0';
  const signature = hmac(`${base['eo-clt-dvidn']}${version}${Object.values(signData).join('')}${auth.cookies.tongyi_sso_ticket_hash}${bodySign}`);
  url.searchParams.set('nonce', nonce);
  url.searchParams.set('timestamp', String(timestamp));
  const headers = {
    ...cookieHeaders(auth), accept: 'text/event-stream',
    'clt-acs-sign': signature, 'clt-acs-reqt': String(timestamp),
    'clt-acs-request-params': Object.keys(signData).join(','), 'clt-acs-bfg': bodySign,
    'clt-acs-caer': 'vrad', 'eo-clt-dvidn': base['eo-clt-dvidn'],
    'eo-clt-sacsft': sac, 'eo-clt-snver': base['eo-clt-snver'],
    'eo-clt-actkn': scene['eo-clt-actkn'], 'eo-clt-acs-ve': version,
    'eo-clt-acs-kp': auth.cookies.tongyi_sso_ticket_hash,
    'x-chat-id': reqId,
    'x-chat-biz': JSON.stringify({ chatId: reqId, agentId: '', enableWebp: '' }),
    'x-platform': 'pc_tongyi', prod_id: 'tongyi',
  };
  return { url, body, headers, reqId };
}

function extractReply(stream) {
  const packets = stream.split(/\r?\n\r?\n/).flatMap((frame) => frame.split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => { try { return JSON.parse(line.slice(5)); } catch { return null; } }))
    .filter(Boolean);
  const errors = packets.filter((packet) => packet.code && packet.code !== 0 || packet.error_code && packet.error_code !== 0);
  if (errors.length) throw new Error(`千问返回错误：${errors.at(-1).error_msg ?? errors.at(-1).code}`);
  const messages = packets.flatMap((packet) => packet.data?.messages ?? []);
  const replies = messages.filter((item) => item.mime_type === 'multi_load/iframe' && typeof item.content === 'string' && item.content);
  if (!replies.length) throw new Error(`千问回复为空；共收到 ${packets.length} 个 SSE 数据包`);
  return replies.at(-1).content;
}

let state;
try {
  const auth = readAuth(false);
  if (deleteSessionStdin) {
    const { session_id: sessionId } = JSON.parse(readFileSync(0, 'utf8'));
    if (typeof sessionId !== 'string' || !/^[a-f0-9]{32}$/.test(sessionId)) throw new Error('会话 ID 无效');
    await deleteSession(auth, sessionId);
    console.log(JSON.stringify({ deleted: sessionId }));
    process.exit(0);
  }
  if (deleteCurrent) {
    state = readState();
    await deleteSession(auth, state.sessionId);
    clearState();
    console.error(`已删除千问会话 ${state.sessionId}`);
  } else {
    let requestPrompt = prompt;
    if (jsonStdin) {
      const request = JSON.parse(readFileSync(0, 'utf8'));
      if (typeof request.prompt !== 'string' || !request.prompt.trim()) throw new Error('缺少提示词');
      requestPrompt = request.prompt;
      if (request.session_id != null && !/^[a-f0-9]{32}$/.test(request.session_id)) throw new Error('会话 ID 无效');
      state = { sessionId: request.session_id || randomUUID().replaceAll('-', ''),
        reqId: '0', topicId: request.topic_id || randomUUID().replaceAll('-', '') };
      if (!request.session_id) console.log(JSON.stringify({ session_id: state.sessionId, topic_id: state.topicId }));
    } else {
      state = continueChat || (!newChat && !deleteAfter && existsSync(STATE_URL)) ? readState() : {
        sessionId: randomUUID().replaceAll('-', ''), reqId: '0', topicId: randomUUID().replaceAll('-', ''),
      };
      if (state.reqId === '0' && !deleteAfter) saveState(state);
      console.error(`会话 ID: ${state.sessionId}`);
    }
    const { url, body, headers, reqId } = await signedRequest(auth, state, requestPrompt);
    const response = await fetch(url, { method: 'POST', headers, body });
    const stream = await response.text();
    if (stream.includes('FAIL_SYS_USER_VALIDATE')) {
      throw new Error(`千问网页应用层拒绝请求（按 HTTP 429 停用本轮来源）：${stream.slice(0, 180)}`);
    }
    if (!response.ok || !response.headers.get('content-type')?.includes('text/event-stream')) {
      throw new Error(`千问聊天请求失败：HTTP ${response.status}, ${stream.slice(0, 300)}`);
    }
    const reply = extractReply(stream);
    if (jsonStdin) console.log(JSON.stringify({ session_id: state.sessionId, topic_id: state.topicId, content: reply }));
    else console.log(reply);
    state.reqId = reqId;
    if (!deleteAfter && !jsonStdin) {
      saveState(state);
      console.error('已保存续聊状态；下次直接运行即可续聊');
    }
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
} finally {
  if (deleteAfter && state) {
    try {
      await deleteSession(readAuth(false), state.sessionId);
      if (existsSync(STATE_URL) && readState().sessionId === state.sessionId) clearState();
      console.error(`已删除本次千问会话 ${state.sessionId}`);
    } catch (error) {
      console.error(`删除会话失败（${state.sessionId}）：${error instanceof Error ? error.message : String(error)}`);
      process.exitCode = 1;
    }
  }
}
