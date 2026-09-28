"""从已登录的千问调试浏览器导出网页请求凭据。需要 websocket-client。"""

import json
import os
import sys
import urllib.request
from pathlib import Path

import websocket


if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    raise SystemExit("用法：python qwen_capture_auth.py <360 浏览器调试端口>")

port = int(sys.argv[1])
tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5))
pages = [tab for tab in tabs if tab.get("type") == "page" and tab.get("url", "").startswith("https://www.qianwen.com/")]
if len(pages) != 1:
    raise SystemExit("请在调试浏览器中打开一个已登录的 https://www.qianwen.com/ 标签页")

socket = websocket.create_connection(pages[0]["webSocketDebuggerUrl"], timeout=15, suppress_origin=True)
serial = 0


def command(method, params=None):
    global serial
    serial += 1
    request_id = serial
    socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
    while True:
        message = json.loads(socket.recv())
        if message.get("id") == request_id:
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result", {})


try:
    expression = """(()=>{let source=performance.getEntriesByType('resource').map(x=>x.name).find(x=>x.startsWith('https://chat2-api.qianwen.com/api/')&&x.includes('?'));if(!source)throw Error('请刷新千问网页，让页面加载会话列表后再运行');let url=new URL(source),storage=JSON.parse(localStorage.getItem('_qk_busc_info_v1qwen_web')||'{}');return JSON.stringify({query:url.searchParams.toString(),storage})})()"""
    evaluated = command("Runtime.evaluate", {"expression": expression, "returnByValue": True})
    if "exceptionDetails" in evaluated:
        raise RuntimeError("未找到千问 API 查询参数；请刷新已登录的千问网页后重试")
    page = json.loads(evaluated["result"]["value"])
    cookies = command("Network.getCookies", {"urls": ["https://www.qianwen.com", "https://chat2.qianwen.com", "https://chat2-api.qianwen.com"]})["cookies"]
    auth = {
        "query": page["query"],
        "storage": page["storage"],
        "cookies": {item["name"]: item["value"] for item in cookies if item.get("domain", "").endswith("qianwen.com")},
    }
    scene = auth["storage"].get("scene_list", {}).get("qwen_chat", {})
    if not scene.get("eo-clt-actkn") or not scene.get("eo-clt-bacsft") or "tongyi_sso_ticket_hash" not in auth["cookies"]:
        raise RuntimeError("网页签名材料尚未初始化；请在千问网页发一条消息后重试")
    target = Path(__file__).with_name(".qwen-web-auth.json")
    with os.fdopen(os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), "w", encoding="utf-8") as stream:
        json.dump(auth, stream, ensure_ascii=False, indent=2)
    print(f"已写入 {target.name}；Cookie {len(auth['cookies'])} 项，可用签名材料 {len(scene['eo-clt-bacsft'])} 项。")
finally:
    socket.close()
