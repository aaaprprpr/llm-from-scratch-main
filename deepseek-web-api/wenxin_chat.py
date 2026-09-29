"""百度文心助手网页快速对话。Python 3 标准库，无浏览器运行时依赖。"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
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


AUTH = ROOT / setting('WENXIN_WEB_AUTH_FILE', '.wenxin-web-auth.json')
STATE = ROOT / '.wenxin-session.json'
TEMPLATE = ROOT / 'wenxin_request_template.json'


def auth():
    try:
        value = json.loads(AUTH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('缺少文心凭据；先运行 python capture_web_auth.py wenxin <调试端口>') from None
    if not value.get('cookies', {}).get('BDUSS'):
        raise RuntimeError('文心登录 Cookie 无效，请重新导出')
    return value


def headers(value):
    return {'Content-Type': 'application/json', 'Cookie': '; '.join(f'{k}={v}' for k, v in value['cookies'].items() if k)}


def post(value, url, body):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8'), headers=headers(value), method='POST')
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'文心 HTTP {error.code}: {error.read(200).decode("utf-8", errors="replace")}') from None


def chat_token(value, prompt):
    request = urllib.request.Request('https://wenxin.baidu.com/', headers={'Cookie': headers(value)['Cookie']})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode('utf-8', errors='replace')
    match = re.search(r'<script type="application/json" name="aiTabFrameBaseData">(.*?)</script>', html, re.S)
    if not match:
        raise RuntimeError('文心首页未提供聊天令牌，请检查登录状态')
    page = json.loads(match.group(1))
    if not page.get('token') or not page.get('lid'):
        raise RuntimeError('文心首页聊天令牌不完整')
    now = str(int(time.time() * 1000))
    lid = str(page['lid'])
    digest = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    text = base64.b64encode(f'{page["token"]}|{digest}|{now}|{lid}'.encode()).decode()
    return f'{text}-{lid}-3'


def read_state():
    try:
        state = json.loads(STATE.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('没有可续聊或删除的文心会话') from None
    if not str(state.get('session_id', '')).isdigit() or not isinstance(state.get('rank'), int):
        raise RuntimeError('文心会话状态文件无效')
    return state


def save_state(state):
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def clear_state(session_id):
    if STATE.exists() and read_state()['session_id'] == session_id:
        STATE.unlink()


def chat(value, prompt, prior):
    body = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    rank = prior['rank'] + 1 if prior else 1
    original = prior['session_id'] if prior else ''
    body['rank'] = rank
    body['message']['query'][0]['data']['text']['query'] = prompt
    body['message']['content']['agentInfo']['params'] = json.dumps({'agt_rk': rank, 'agt_sess_cnt': 1}, separators=(',', ':'))
    search = body['message']['searchInfo']
    search['ori_lid'] = original
    search['re_rank'] = str(rank)
    search['chatParams']['chat_token'] = chat_token(value, prompt)
    search['usedModel']['modelFunction'] = {'deepSearch': '0', 'thinkMode': '0'}
    search['capsuleSelectMode'] = 'fast'
    answer = []
    state = {'session_id': original, 'rank': rank}
    event = None
    data_lines = []

    def handle():
        nonlocal event, data_lines
        if not data_lines:
            event = None
            return
        try:
            packet = json.loads('\n'.join(data_lines))
        except ValueError:
            event, data_lines = None, []
            return
        if packet.get('status', 0) != 0:
            raise RuntimeError(f'文心返回业务错误：{packet.get("status")}')
        if event == 'message':
            if packet.get('sessionId'):
                state['session_id'] = str(packet['sessionId'])
            generator = packet.get('data', {}).get('message', {}).get('content', {}).get('generator', {})
            if generator.get('component') == 'markdown-yiyan':
                part = generator.get('data', {}).get('value', '')
                if part:
                    print(part, end='', flush=True)
                    answer.append(part)
        event, data_lines = None, []

    with post(value, 'https://wenxin.baidu.com/aichat/api/conversation', body) as response:
        if 'text/event-stream' not in response.headers.get('content-type', ''):
            raise RuntimeError('文心没有返回聊天流')
        for raw in response:
            line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
            if not line:
                handle()
            elif line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                data_lines.append(line[5:].lstrip())
        handle()
    if answer:
        print(flush=True)
    if not state['session_id'] or not answer:
        raise RuntimeError('文心响应缺少会话 ID 或正文')
    return state


def delete_chat(value, session_id):
    body = {'req_type': 1, 'ori_lid_list': [{'ori_lid': session_id, 'source': 'csaitab'}]}
    with post(value, 'https://chat.baidu.com/csaitab/history/delete', body) as response:
        payload = json.load(response)
        if response.status != 200 or payload.get('status', 0) != 0:
            raise RuntimeError(f'文心删除失败：{str(payload)[:180]}')


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
    if (not deleting and not prompt) or (deleting and (prompt or args.new or args.continuing or args.delete_after)):
        parser.error('请提供消息，或单独使用 --delete-current')
    value = auth()
    if deleting:
        session_id = args.delete_id or read_state()['session_id']
        if not str(session_id).isdigit():
            parser.error('会话 ID 必须为数字')
        delete_chat(value, session_id)
        clear_state(session_id)
        print(f'已删除文心会话 {session_id}', file=sys.stderr)
        return
    prior = read_state() if args.continuing or (not args.new and not args.delete_after and STATE.exists()) else None
    created = None
    try:
        created = chat(value, prompt, prior)
        print(f'会话 ID: {created["session_id"]}', file=sys.stderr)
        if not args.delete_after:
            save_state(created)
    finally:
        if args.delete_after and created:
            delete_chat(value, created['session_id'])
            clear_state(created['session_id'])
            print(f'已删除本次文心会话 {created["session_id"]}', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
