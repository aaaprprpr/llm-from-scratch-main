"""从已登录的腾讯元宝调试浏览器导出 Cookie 和网页签名。需要 websocket-client。"""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import websocket

if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    raise SystemExit('用法：python yuanbao_capture_auth.py <360 浏览器调试端口>')

port = int(sys.argv[1])
tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=5))
pages = [tab for tab in tabs if tab.get('type') == 'page' and urlsplit(tab.get('url', '')).hostname == 'yuanbao.tencent.com']
if len(pages) != 1:
    raise SystemExit('请在调试浏览器中打开一个已登录的 https://yuanbao.tencent.com/ 标签页')

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
    expression = """JSON.stringify((()=>{let req;window.webpackChunk_N_E.push([['codex-auth-export-'+Date.now()],{},r=>req=r]);let m=req(77004),q=m.I5(m.PU),h=q?.getLocalQimei36()?.h38,n=Date.now(),text=`h38=${h}&timestamp=${n}&platform=web`,key=q?.getUSKeySync('7800385',h,text);return {h38:h,timestamp:n,plain:text,uskey:key?encodeURIComponent(key):'',hy92:m.mD(q),hy93:m.BQ(q),userAgent:navigator.userAgent}})())"""
    result = command('Runtime.evaluate', {'expression': expression, 'returnByValue': True})
    if 'exceptionDetails' in result:
        raise RuntimeError('元宝网页签名模块不可用：' + result['exceptionDetails'].get('text', '未知错误'))
    signed = json.loads(result['result']['value'])
    cookies = command('Network.getCookies', {'urls': ['https://yuanbao.tencent.com']})['cookies']
finally:
    socket.close()

if len(signed.get('h38', '')) != 38 or not signed.get('uskey') or not signed.get('hy92') or not signed.get('hy93'):
    raise RuntimeError('元宝签名信息不完整，请确认已登录并刷新页面')
auth = {
    'cookies': {item['name']: item['value'] for item in cookies if item.get('domain', '').endswith('tencent.com')},
    'signature': {
        'uskey': signed['uskey'],
        'bus_params_md5': hashlib.md5(signed['plain'].encode()).hexdigest(),
        'timestamp': str(signed['timestamp']),
        'hy92': signed['hy92'],
        'hy93': signed['hy93'],
        'user_agent': signed['userAgent'],
    },
}
if not auth['cookies'].get('hy_token'):
    raise RuntimeError('元宝登录 Cookie 不完整，请重新登录')
target = Path(__file__).with_name('.yuanbao-web-auth.json')
with os.fdopen(os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), 'w', encoding='utf-8') as stream:
    json.dump(auth, stream, ensure_ascii=False, indent=2)
print(f'已更新 {target.name}，Cookie {len(auth["cookies"])} 项。聊天脚本运行时不需要浏览器。')
