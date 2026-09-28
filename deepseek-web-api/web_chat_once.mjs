import { deepseekHash } from './pow23.mjs';
import { readFileSync, writeFileSync, unlinkSync, existsSync } from 'node:fs';

function tokenFromDotEnv() {
  let text;
  try {
    text = readFileSync(new URL('.env', import.meta.url), 'utf8');
  } catch (error) {
    if (error?.code === 'ENOENT') return undefined;
    throw error;
  }
  const line = text.split(/\r?\n/).find((item) => /^\s*DEEPSEEK_WEB_TOKEN\s*=/.test(item));
  if (!line) return undefined;
  const value = line.slice(line.indexOf('=') + 1).trim();
  if (value.startsWith('"')) return JSON.parse(value);
  if (value.startsWith("'") && value.endsWith("'")) return value.slice(1, -1);
  return value;
}

const BASE = 'https://chat.deepseek.com';
const COMPLETION_PATH = '/api/v0/chat/completion';
const STATE_URL = new URL('.deepseek-session.json', import.meta.url);
const token = (process.env.DEEPSEEK_WEB_TOKEN || tokenFromDotEnv())?.trim();
const args = process.argv.slice(2);
const continueChat = args.includes('--continue');
const newChat = args.includes('--new');
const deleteCurrent = args.includes('--delete-current');
const deleteAfter = args.includes('--delete-after');
const stdinPrompt = args.includes('--stdin');
const jsonStdin = args.includes('--json-stdin');
const deleteSessionStdin = args.includes('--delete-session-stdin');
const prompt = args.filter((arg) => !['--continue', '--new', '--delete-current', '--delete-after', '--stdin', '--json-stdin', '--delete-session-stdin'].includes(arg)).join(' ').trim();

if (!token || (!deleteCurrent && !stdinPrompt && !jsonStdin && !deleteSessionStdin && !prompt) ||
    (newChat && continueChat) ||
    ((stdinPrompt || jsonStdin || deleteSessionStdin) && (prompt || continueChat || newChat || deleteCurrent || deleteAfter)) ||
    (deleteCurrent && (prompt || continueChat || newChat || deleteAfter))) {
  console.error('用法：node web_chat_once.mjs [--new | --continue | --delete-after] "你好"；或 node web_chat_once.mjs --delete-current');
  process.exit(1);
}

function readState() {
  try {
    const state = JSON.parse(readFileSync(STATE_URL, 'utf8'));
    if (typeof state.sessionId !== 'string' || !state.sessionId) throw new Error('会话状态文件无效');
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

const commonHeaders = {
  Authorization: `Bearer ${token}`,
  'Content-Type': 'application/json',
};

async function postJson(path, body) {
  const response = await fetch(BASE + path, {
    method: 'POST',
    headers: commonHeaders,
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok || payload.code !== 0 || payload.data?.biz_code !== 0) {
    throw new Error(`${path}: HTTP ${response.status}, code=${payload.code}, biz_code=${payload.data?.biz_code}, msg=${payload.msg ?? payload.data?.biz_msg ?? ''}`);
  }
  return payload.data.biz_data;
}

function solveChallenge(challenge) {
  if (challenge.algorithm !== 'DeepSeekHashV1') {
    throw new Error(`未知 PoW 算法：${challenge.algorithm}`);
  }
  const { salt, expire_at: expireAt, difficulty, challenge: target } = challenge;
  if (!Number.isSafeInteger(difficulty) || difficulty <= 0) {
    throw new Error('无效的 PoW difficulty');
  }
  const prefix = `${salt}_${expireAt}_`;
  for (let answer = 0; answer < difficulty; answer++) {
    if (deepseekHash(prefix + answer) === target) return answer;
  }
  throw new Error('PoW 未找到答案；请检查网页版本或重试');
}

function printSseFrame(frame, state) {
  const lines = frame.split(/\r?\n/);
  let event = '';
  const data = [];
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
  }
  state.events[event || 'message'] = (state.events[event || 'message'] ?? 0) + 1;
  if (event === 'close' || !data.length) return;
  let packet;
  try { packet = JSON.parse(data.join('\n')); } catch { state.invalidFrames++; return; }
  if (event === 'hint') {
    const detail = packet.v && typeof packet.v === 'object' ? packet.v : {};
    const message = packet.msg ?? detail.msg ?? detail.message ?? packet.content ??
      (typeof packet.v === 'string' ? packet.v : '');
    state.hints.push({ keys: Object.keys(packet), detailKeys: Object.keys(detail),
      code: packet.code ?? detail.code ?? null, finishReason: packet.finish_reason ?? null,
      clearResponse: packet.clear_response ?? null,
      message: typeof message === 'string' ? message.slice(0, 120) : '' });
  }
  if (event === 'ready') {
    state.responseMessageId = packet.response_message_id;
    return;
  }
  const initial = packet.v?.response?.fragments;
  if (Array.isArray(initial)) {
    for (const fragment of initial) {
      state.fragmentTypes[fragment.type || 'unknown'] = (state.fragmentTypes[fragment.type || 'unknown'] ?? 0) + 1;
      if (fragment.type === 'RESPONSE' && fragment.content) {
        state.content += fragment.content;
        if (!state.structured) process.stdout.write(fragment.content);
      }
    }
  }
  if (packet.p) state.lastPath = packet.p;
  if (state.lastPath?.endsWith('/content') && typeof packet.v === 'string' && (!packet.o || packet.o === 'APPEND')) {
    state.content += packet.v;
    if (!state.structured) process.stdout.write(packet.v);
  }
}

async function printStream(response, structured = false) {
  const contentType = response.headers.get('content-type') ?? '';
  if (!response.ok || !contentType.includes('text/event-stream')) {
    const error = await response.text();
    throw new Error(`completion: HTTP ${response.status}, ${error.slice(0, 500)}`);
  }
  const state = { lastPath: null, responseMessageId: null, content: '', structured,
    events: {}, fragmentTypes: {}, invalidFrames: 0, hints: [] };
  const decoder = new TextDecoder();
  let buffer = '';
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true }).replace(/\r\n/g, '\n');
    let boundary;
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      printSseFrame(buffer.slice(0, boundary), state);
      buffer = buffer.slice(boundary + 2);
    }
  }
  if (buffer.trim()) printSseFrame(buffer, state);
  if (!structured) process.stdout.write('\n');
  if (!state.content) {
    const limit = state.hints.find((hint) => ['parallel_chat_limit', 'rate_limit_reached'].includes(hint.finishReason));
    if (limit) throw new Error(`${limit.finishReason}: ${limit.message}`);
    throw new Error(`回复流没有正文（事件 ${JSON.stringify(state.events)}，` +
      `片段 ${JSON.stringify(state.fragmentTypes)}，提示 ${JSON.stringify(state.hints)}，无效帧 ${state.invalidFrames}）`);
  }
  if (state.responseMessageId == null && !structured) throw new Error('回复流未给出 response_message_id，未更新续聊状态');
  return state;
}

let activeSession;
try {
  if (deleteSessionStdin) {
    const { session_id: sessionId } = JSON.parse(readFileSync(0, 'utf8'));
    if (typeof sessionId !== 'string' || !sessionId) throw new Error('缺少会话 ID');
    await postJson('/api/v0/chat_session/delete', { chat_session_id: sessionId });
    console.log(JSON.stringify({ deleted: sessionId }));
    process.exit(0);
  }
  if (deleteCurrent) {
    const state = readState();
    await postJson('/api/v0/chat_session/delete', { chat_session_id: state.sessionId });
    clearState();
    console.error(`已删除会话 ${state.sessionId}`);
    process.exit(0);
  }
  let requestPrompt = stdinPrompt ? readFileSync(0, 'utf8').trim() : prompt;
  if (!requestPrompt && stdinPrompt) throw new Error('缺少提示词');
  if (jsonStdin) {
    const request = JSON.parse(readFileSync(0, 'utf8'));
    if (typeof request.prompt !== 'string' || !request.prompt.trim()) throw new Error('缺少提示词');
    if (request.session_id != null && (typeof request.session_id !== 'string' || !request.session_id)) throw new Error('会话 ID 无效');
    requestPrompt = request.prompt;
    if (request.session_id) activeSession = { sessionId: request.session_id, parentMessageId: null };
  }
  if (!jsonStdin && (continueChat || (!newChat && !deleteAfter && existsSync(STATE_URL)))) {
    activeSession = readState();
  } else if (!activeSession) {
    const session = await postJson('/api/v0/chat_session/create', {});
    activeSession = { sessionId: session.id, parentMessageId: null };
    if (jsonStdin) console.log(JSON.stringify({ session_id: activeSession.sessionId }));
    else if (!deleteAfter) saveState(activeSession);
  }
  if (!jsonStdin) console.error(`会话 ID: ${activeSession.sessionId}`);
  const { challenge } = await postJson('/api/v0/chat/create_pow_challenge', {
    target_path: COMPLETION_PATH,
  });
  const answer = solveChallenge(challenge);
  const pow = {
    algorithm: challenge.algorithm,
    challenge: challenge.challenge,
    salt: challenge.salt,
    answer,
    signature: challenge.signature,
    target_path: COMPLETION_PATH,
  };
  const response = await fetch(BASE + COMPLETION_PATH, {
    method: 'POST',
    headers: {
      ...commonHeaders,
      'X-DS-PoW-Response': Buffer.from(JSON.stringify(pow)).toString('base64'),
    },
    body: JSON.stringify({
      chat_session_id: activeSession.sessionId,
      parent_message_id: activeSession.parentMessageId,
      model_type: activeSession.parentMessageId == null ? 'default' : null,
      prompt: requestPrompt,
      ref_file_ids: [],
      thinking_enabled: false,
      search_enabled: false,
      action: null,
      preempt: false,
    }),
  });
  const reply = await printStream(response, jsonStdin);
  activeSession.parentMessageId = reply.responseMessageId;
  if (jsonStdin) {
    console.log(JSON.stringify({ session_id: activeSession.sessionId, content: reply.content,
      response_message_id: reply.responseMessageId }));
  } else if (!deleteAfter) {
    saveState(activeSession);
    console.error(`已保存续聊状态；下次直接运行即可续聊（上一轮助手消息 ID: ${activeSession.parentMessageId}）`);
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
} finally {
  if (deleteAfter && activeSession) {
    try {
      await postJson('/api/v0/chat_session/delete', { chat_session_id: activeSession.sessionId });
      if (existsSync(STATE_URL) && readState().sessionId === activeSession.sessionId) clearState();
      console.error(`已删除本次会话 ${activeSession.sessionId}`);
    } catch (error) {
      console.error(`删除会话失败（${activeSession.sessionId}）：${error instanceof Error ? error.message : String(error)}`);
      process.exitCode = 1;
    }
  }
}
