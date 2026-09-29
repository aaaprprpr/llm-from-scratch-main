"""从已登录的智谱清言调试浏览器导出运行凭据。需要 websocket-client。"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import websocket

if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    raise SystemExit('用法：python chatglm_capture_auth.py <360 浏览器调试端口>')

port = int(sys.argv[1])
tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=5))
pages = [tab for tab in tabs if tab.get('type') == 'page' and urlsplit(tab.get('url', '')).hostname == 'chatglm.cn']
if len(pages) != 1:
    raise SystemExit('请在调试浏览器中打开一个已登录的 https://chatglm.cn/ 标签页')

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
    cookies = command('Network.getCookies', {'urls': ['https://chatglm.cn']})['cookies']
    result = command('Runtime.evaluate', {'expression': "JSON.stringify({device_id:localStorage.getItem('chatglm-deid'),trial_group:localStorage.getItem('trialGroup')||''})", 'returnByValue': True})
    storage = json.loads(result['result']['value'])
finally:
    socket.close()

auth = {'cookies': {item['name']: item['value'] for item in cookies if item.get('domain', '').endswith('chatglm.cn')}, **storage}
if not auth['cookies'].get('chatglm_token') or not auth.get('device_id'):
    raise RuntimeError('智谱登录凭据不完整，请确认已登录并刷新页面')
target = Path(__file__).with_name('.chatglm-web-auth.json')
with os.fdopen(os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), 'w', encoding='utf-8') as stream:
    json.dump(auth, stream, ensure_ascii=False, indent=2)
print(f'已更新 {target.name}，Cookie {len(auth["cookies"])} 项。运行聊天脚本不需要浏览器。')
