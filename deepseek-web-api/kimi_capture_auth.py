"""从已登录的 Kimi 调试浏览器导出访问和刷新令牌。需要 websocket-client。"""

import json
import os
import sys
import urllib.request
from pathlib import Path

import websocket


if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    raise SystemExit('用法：python kimi_capture_auth.py <360 浏览器调试端口>')

port = int(sys.argv[1])
tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=5))
pages = [tab for tab in tabs if tab.get('type') == 'page' and tab.get('url', '').startswith('https://www.kimi.com/')]
if len(pages) != 1:
    raise SystemExit('请在调试浏览器中打开一个已登录的 https://www.kimi.com/ 标签页')

socket = websocket.create_connection(pages[0]['webSocketDebuggerUrl'], timeout=15, suppress_origin=True)
try:
    socket.send(json.dumps({'id': 1, 'method': 'Runtime.evaluate', 'params': {
        'expression': "JSON.stringify({accessToken:localStorage.getItem('access_token'),refreshToken:localStorage.getItem('refresh_token')})",
        'returnByValue': True,
    }}))
    while True:
        message = json.loads(socket.recv())
        if message.get('id') == 1:
            break
    if 'error' in message or 'exceptionDetails' in message.get('result', {}):
        raise RuntimeError('读取 Kimi 登录令牌失败')
    values = json.loads(message['result']['result']['value'])
    if not values.get('accessToken') or not values.get('refreshToken'):
        raise RuntimeError('Kimi 登录令牌为空，请在网页重新登录')
    target = Path(__file__).with_name('.kimi-web-auth.json')
    with os.fdopen(os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), 'w', encoding='utf-8') as stream:
        json.dump(values, stream, ensure_ascii=False, indent=2)
    print(f'已更新 {target.name}；运行时不需要浏览器。')
finally:
    socket.close()
