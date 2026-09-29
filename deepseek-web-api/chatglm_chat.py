"""智谱清言网页 GLM-Flash 快速模式直连聊天。运行时仅需 Python 3 标准库。"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def setting(name, default):
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in (ROOT / '.env').read_text(encoding='utf-8').splitlines():
            if line.lstrip().startswith(name + '='):
                value = line.split('=', 1)[1].strip()
                return json.loads(value) if value.startswith('"') else value.strip("'")
    except FileNotFoundError:
        pass
    return default


AUTH = ROOT / setting('CHATGLM_WEB_AUTH_FILE', '.chatglm-web-auth.json')
STATE = ROOT / '.chatglm-session.json'
ASSISTANT_ID = '65940acff94777010aa6b796'
BASE = 'https://chatglm.cn'


def auth():
    try:
        value = json.loads(AUTH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('缺少智谱凭据；先运行 python chatglm_capture_auth.py <调试端口>') from None
    if not value.get('cookies', {}).get('chatglm_token') or not value.get('device_id'):
        raise RuntimeError('智谱登录凭据不完整，请重新导出')
    return value


def signed_headers(value):
    raw = str(int(time.time() * 1000))
    digits = [int(char) for char in raw]
    checksum = (sum(digits) - digits[-2]) % 10
    timestamp = raw[:-2] + str(checksum) + raw[-1]
    nonce = uuid.uuid4().hex
    signature = hashlib.md5(f'{timestamp}-{nonce}-8a1317a7468aa3ad86e997d08f3f31cb'.encode()).hexdigest()
    return {
        'Authorization': 'Bearer ' + value['cookies']['chatglm_token'],
        'Cookie': '; '.join(f'{k}={v}' for k, v in value['cookies'].items() if k),
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'App-Name': 'chatglm',
        'X-Lang': 'zh',
        'X-Device-Id': value['device_id'],
        'X-App-Platform': 'pc',
        'X-App-Version': '0.0.1',
        'X-App-fr': 'default',
        'X-Request-Id': uuid.uuid4().hex,
        'X-Exp-Groups': value.get('trial_group', ''),
        'X-Device-Model': '',
        'X-Device-Brand': '',
        'X-Timestamp': timestamp,
        'X-Nonce': nonce,
        'X-Sign': signature,
    }


def post(value, path, body):
    data = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data, headers=signed_headers(value), method='POST')
    try:
        return urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'智谱 HTTP {error.code}: {error.read(300).decode("utf-8", errors="replace")}') from None


def read_state():
    try:
        state = json.loads(STATE.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('没有可续聊或删除的智谱会话') from None
    if not state.get('conversation_id'):
        raise RuntimeError('智谱会话状态文件无效')
    return state


def delete_chat(value, conversation_id):
    body = {'conversation_ids': [conversation_id], 'assistant_id': ASSISTANT_ID}
    with post(value, '/chatglm/mainchat-api/conversation/bulk_delete', body) as response:
        result = json.load(response)
    if result.get('status') not in (None, 0, 200) or result.get('code') not in (None, 0, 200):
        raise RuntimeError(f'智谱删除失败：{str(result)[:220]}')


def chat(value, prompt, conversation_id):
    body = {
        'assistant_id': ASSISTANT_ID,
        'conversation_id': conversation_id,
        'project_id': '',
        'chat_type': 'user_chat',
        'meta_data': {
            'cogview': {'rm_label_watermark': False},
            'is_test': False,
            'input_question_type': 'xxxx',
            'channel': '',
            'draft_id': '',
            'chat_mode': '',
            'reasoning_effort': 'low',
            'selected_model': 'glm-5.3-flash',
            'is_networking': False,
            'quote_log_id': '',
            'platform': 'pc',
        },
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
    }
    answer = ''
    new_id = conversation_id
    finished = False
    with post(value, '/chatglm/backend-api/assistant/stream', body) as response:
        if 'text/event-stream' not in response.headers.get('content-type', ''):
            raise RuntimeError('智谱没有返回聊天流')
        for raw in response:
            line = raw.decode('utf-8', errors='replace').strip()
            if not line.startswith('data:'):
                continue
            try:
                packet = json.loads(line[5:].strip())
            except ValueError:
                continue
            if packet.get('conversation_id'):
                new_id = packet['conversation_id']
            if packet.get('last_error'):
                raise RuntimeError(f'智谱聊天失败：{str(packet["last_error"])[:220]}')
            for part in packet.get('parts', []):
                for content in part.get('content', []):
                    if content.get('type') != 'text':
                        continue
                    text = content.get('text', '')
                    if text.startswith(answer):
                        delta = text[len(answer):]
                        answer = text
                    else:
                        delta = text
                        answer += text
                    if delta:
                        print(delta, end='', flush=True)
            finished = packet.get('status') == 'finish' or finished
    if answer:
        print(flush=True)
    if not finished or not new_id or not answer:
        raise RuntimeError('智谱聊天流不完整，未保存会话状态')
    return {'conversation_id': new_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--new', action='store_true')
    group.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--delete-after', action='store_true')
    parser.add_argument('--delete-current', action='store_true')
    parser.add_argument('--delete-id', metavar='ID')
    parser.add_argument('prompt', nargs='*')
    args = parser.parse_args()
    prompt = ' '.join(args.prompt).strip()
    deleting = args.delete_current or args.delete_id
    if deleting:
        if prompt or args.new or args.continuing or args.delete_after:
            parser.error('删除时不要附带聊天消息')
    elif not prompt:
        parser.error('请提供消息')
    value = auth()
    if deleting:
        cid = args.delete_id or read_state()['conversation_id']
        delete_chat(value, cid)
        if STATE.exists() and read_state()['conversation_id'] == cid:
            STATE.unlink()
        print(f'已删除智谱会话 {cid}', file=sys.stderr)
        return
    prior = read_state() if args.continuing or (not args.new and not args.delete_after and STATE.exists()) else None
    state = None
    try:
        state = chat(value, prompt, prior['conversation_id'] if prior else '')
        print(f'会话 ID: {state["conversation_id"]}', file=sys.stderr)
        if not args.delete_after:
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    finally:
        if args.delete_after and state:
            delete_chat(value, state['conversation_id'])
            print(f'已删除本次智谱会话 {state["conversation_id"]}', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
