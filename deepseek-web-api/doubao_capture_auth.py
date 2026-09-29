"""从已登录的豆包调试浏览器导出 Cookie 和公共查询参数。需要 websocket-client。"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import websocket


if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    raise SystemExit('用法：python doubao_capture_auth.py <360 浏览器调试端口>')

port = int(sys.argv[1])
tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=5))
pages = [tab for tab in tabs if tab.get('type') == 'page' and tab.get('url', '').startswith('https://www.doubao.com/')]
if len(pages) != 1:
    raise SystemExit('请在调试浏览器中打开一个已登录的 https://www.doubao.com/ 标签页')

socket = websocket.create_connection(pages[0]['webSocketDebuggerUrl'], timeout=15, suppress_origin=True)
serial = 0


def command(method, params=None):
    global serial
    serial += 1
    current = serial
    socket.send(json.dumps({'id': current, 'method': method, 'params': params or {}}))
    while True:
        message = json.loads(socket.recv())
        if message.get('id') == current:
            if 'error' in message:
                raise RuntimeError(message['error'])
            return message.get('result', {})


try:
    expression = """JSON.stringify(performance.getEntriesByType('resource').map(x=>x.name).filter(x=>x.startsWith('https://www.doubao.com/')&&x.includes('?')).reverse().find(x=>new URL(x).pathname==='/chat/completion')||'')"""
    evaluated = command('Runtime.evaluate', {'expression': expression, 'returnByValue': True})
    source = json.loads(evaluated['result']['value'])
    target = Path(__file__).with_name('.doubao-web-auth.json')
    if not source and target.exists():
        query = json.loads(target.read_text(encoding='utf-8')).get('query')
    elif source:
        query = urlencode([(key, value) for key, value in parse_qsl(urlsplit(source).query) if key not in ('a_bogus', 'msToken')])
    else:
        raise RuntimeError('未找到豆包聊天查询参数；请在网页发送一条消息后重新运行')
    cookies = command('Network.getCookies', {'urls': ['https://www.doubao.com', 'https://doubao.com']})['cookies']
    auth = {'query': query, 'cookies': {item['name']: item['value'] for item in cookies if item.get('domain', '').endswith('doubao.com')}}
    if not auth['cookies']:
        raise RuntimeError('未找到豆包登录 Cookie，请在网页重新登录')
    with os.fdopen(os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), 'w', encoding='utf-8') as stream:
        json.dump(auth, stream, ensure_ascii=False, indent=2)
    print(f'已更新 {target.name}；Cookie {len(auth["cookies"])} 项。运行时不需要浏览器。')
finally:
    socket.close()
