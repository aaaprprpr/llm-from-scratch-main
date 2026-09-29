"""从已登录的调试浏览器导出文心或讯飞的 Cookie。仅导出时需要 websocket-client。"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import websocket

if len(sys.argv) != 3 or sys.argv[1] not in ('wenxin', 'spark') or not sys.argv[2].isdigit():
    raise SystemExit('用法：python capture_web_auth.py {wenxin|spark} <360 浏览器调试端口>')

site, port = sys.argv[1], int(sys.argv[2])
hosts = {'wenxin': 'wenxin.baidu.com', 'spark': 'spark.xfyun.cn'}
urls = {'wenxin': ['https://wenxin.baidu.com', 'https://chat.baidu.com', 'https://baidu.com'],
        'spark': ['https://spark.xfyun.cn']}
host = hosts[site]
tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=5))
pages = [tab for tab in tabs if tab.get('type') == 'page' and urlsplit(tab.get('url', '')).hostname == host]
if len(pages) != 1:
    raise SystemExit(f'请在调试浏览器中打开一个已登录的 https://{host}/ 标签页')

socket = websocket.create_connection(pages[0]['webSocketDebuggerUrl'], timeout=15, suppress_origin=True)
try:
    socket.send(json.dumps({'id': 1, 'method': 'Network.getCookies', 'params': {'urls': urls[site]}}))
    while True:
        message = json.loads(socket.recv())
        if message.get('id') == 1:
            if 'error' in message:
                raise RuntimeError(message['error'])
            cookies = message['result']['cookies']
            break
finally:
    socket.close()

if site == 'wenxin':
    cookies = [item for item in cookies if item.get('domain', '').endswith('baidu.com')]
    if not any(item['name'] == 'BDUSS' for item in cookies):
        raise RuntimeError('文心登录 Cookie 不完整，请确认已登录')
else:
    cookies = [item for item in cookies if item.get('domain', '').endswith('xfyun.cn')]
    if not cookies:
        raise RuntimeError('讯飞登录 Cookie 为空，请确认已登录')

target = Path(__file__).with_name(f'.{site}-web-auth.json')
with os.fdopen(os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), 'w', encoding='utf-8') as stream:
    json.dump({'cookies': {item['name']: item['value'] for item in cookies}}, stream, ensure_ascii=False, indent=2)
print(f'已更新 {target.name}，Cookie {len(cookies)} 项。运行聊天脚本不需要浏览器。')
