import { deepseekHash } from './pow23.mjs';

const BASE = 'https://chat.deepseek.com';
const COMPLETION_PATH = '/api/v0/chat/completion';
const token = process.env.DEEPSEEK_WEB_TOKEN?.trim();
const prompt = process.argv.slice(2).join(' ').trim();

if (!token || !prompt) {
  console.error('用法：设置 DEEPSEEK_WEB_TOKEN，然后运行 node web_chat_once.mjs "你好"');
  process.exit(1);
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
  if (event === 'close' || !data.length) return;
  let packet;
  try { packet = JSON.parse(data.join('\n')); } catch { return; }
  if (event === 'ready') {
    state.responseMessageId = packet.response_message_id;
    return;
  }
  const initial = packet.v?.response?.fragments;
  if (Array.isArray(initial)) {
    for (const fragment of initial) {
      if (fragment.type === 'RESPONSE' && fragment.content) process.stdout.write(fragment.content);
    }
  }
  if (packet.p) state.lastPath = packet.p;
  if (state.lastPath?.endsWith('/content') && typeof packet.v === 'string' && (!packet.o || packet.o === 'APPEND')) {
    process.stdout.write(packet.v);
  }
}

async function printStream(response) {
  const contentType = response.headers.get('content-type') ?? '';
  if (!response.ok || !contentType.includes('text/event-stream')) {
    const error = await response.text();
    throw new Error(`completion: HTTP ${response.status}, ${error.slice(0, 500)}`);
  }
  const state = { lastPath: null, responseMessageId: null };
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
  process.stdout.write('\n');
  return state.responseMessageId;
}

try {
  const session = await postJson('/api/v0/chat_session/create', {});
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
      chat_session_id: session.id,
      parent_message_id: null,
      model_type: 'default',
      prompt,
      ref_file_ids: [],
      thinking_enabled: false,
      search_enabled: false,
      action: null,
      preempt: false,
    }),
  });
  await printStream(response);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
