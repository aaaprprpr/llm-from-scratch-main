"""讯飞星火网页 Fast 模式直连聊天。运行时仅需 Python 3 标准库。"""

import argparse
import base64
import json
import os
import re
import sys
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


AUTH = ROOT / setting('SPARK_WEB_AUTH_FILE', '.spark-web-auth.json')
STATE = ROOT / '.spark-session.json'
BASE = 'https://spark.xfyun.cn'


def auth():
    try:
        value = json.loads(AUTH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('缺少讯飞凭据；先运行 python capture_web_auth.py spark <调试端口>') from None
    if not value.get('cookies'):
        raise RuntimeError('讯飞 Cookie 为空，请重新导出')
    return value


def cookie_header(value):
    return '; '.join(f'{k}={v}' for k, v in value['cookies'].items() if k)


def request(value, path, data, content_type):
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': content_type, 'Cookie': cookie_header(value)}, method='POST')
    try:
        return urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'讯飞 HTTP {error.code}: {error.read(300).decode("utf-8", errors="replace")}') from None


def post_json(value, path, payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    with request(value, path, data, 'application/json') as response:
        result = json.load(response)
    if result.get('code') != 0 or result.get('flag') is False:
        raise RuntimeError(f'讯飞接口失败：{str(result)[:250]}')
    return result


def create_chat(value):
    result = post_json(value, '/iflygpt/u/chat-list/v1/create-chat-list', {'showType': '2'})
    chat_id = result.get('data', {}).get('id')
    if not str(chat_id).isdigit():
        raise RuntimeError('讯飞未返回会话 ID')
    return str(chat_id)


def delete_chat(value, chat_id):
    post_json(value, '/iflygpt/u/chat-list/v1/batch-delete-chat-list', {'chatListIds': [int(chat_id)]})


def multipart(fields):
    boundary = '----SparkChat' + uuid.uuid4().hex
    chunks = []
    for key, value in fields:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n')
    chunks.append(f'--{boundary}--\r\n')
    return ''.join(chunks).encode('utf-8'), f'multipart/form-data; boundary={boundary}'


def chat(value, prompt, chat_id, sid=None):
    fields = [
        ('fd', str(uuid.uuid4().int % 900000 + 100000)),
        ('isBot', '0'),
        ('clientType', '1'),
        ('text', prompt),
        ('chatId', chat_id),
        ('options', json.dumps({'chatOption': {'thinkPattern': 'fast_think', 'routeVersion': 'legacy'}}, separators=(',', ':'))),
    ]
    if sid:
        fields.append(('sid', sid))
    else:
        fields.append(('firstChat', 'true'))
    body, content_type = multipart(fields)
    answer = []
    next_sid = None
    saw_end = False
    with request(value, '/iflygpt-chat/u/chat_message/chat', body, content_type) as response:
        for raw in response:
            line = raw.decode('utf-8', errors='replace').strip()
            if not line.startswith('data:'):
                continue
            content = line[5:]
            if content == '<end>':
                saw_end = True
                continue
            if content.endswith('<sid>'):
                next_sid = content[:-5]
                continue
            try:
                piece = base64.b64decode(content, validate=True).decode('utf-8')
            except (ValueError, UnicodeDecodeError):
                continue
            # The site may append a metadata block after the visible answer.
            piece = re.split(r'```memoryExtraction\b', piece, maxsplit=1)[0]
            if piece:
                answer.append(piece)
                print(piece, end='', flush=True)
    if answer:
        print(flush=True)
    if not saw_end or not answer or not next_sid:
        raise RuntimeError('讯飞聊天流不完整，未保存会话状态')
    return {'chat_id': chat_id, 'sid': next_sid}


def read_state():
    try:
        state = json.loads(STATE.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('没有可续聊或删除的讯飞会话') from None
    if not str(state.get('chat_id', '')).isdigit() or not state.get('sid'):
        raise RuntimeError('讯飞会话状态文件无效')
    return state


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
    if args.delete_current or args.delete_id:
        if prompt or args.new or args.continuing or args.delete_after:
            parser.error('删除时不要附带聊天消息')
    elif not prompt:
        parser.error('请提供消息')
    value = auth()
    if args.delete_current or args.delete_id:
        chat_id = args.delete_id or read_state()['chat_id']
        if not str(chat_id).isdigit():
            parser.error('会话 ID 必须为数字')
        delete_chat(value, chat_id)
        if STATE.exists() and read_state()['chat_id'] == chat_id:
            STATE.unlink()
        print(f'已删除讯飞会话 {chat_id}', file=sys.stderr)
        return
    prior = read_state() if args.continuing or (not args.new and not args.delete_after and STATE.exists()) else None
    chat_id = prior['chat_id'] if prior else create_chat(value)
    state = None
    try:
        state = chat(value, prompt, chat_id, prior['sid'] if prior else None)
        print(f'会话 ID: {chat_id}', file=sys.stderr)
        if not args.delete_after:
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    finally:
        if args.delete_after:
            delete_chat(value, chat_id)
            print(f'已删除本次讯飞会话 {chat_id}', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
