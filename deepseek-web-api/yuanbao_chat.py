"""腾讯元宝网页快速回答直连聊天。运行时仅需 Python 3 标准库。"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = 'https://yuanbao.tencent.com'
AGENT_ID = 'naQivTmsDa'


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


AUTH = ROOT / setting('YUANBAO_WEB_AUTH_FILE', '.yuanbao-web-auth.json')
STATE = ROOT / '.yuanbao-session.json'


def auth():
    try:
        value = json.loads(AUTH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('缺少元宝凭据；先运行 python yuanbao_capture_auth.py <调试端口>') from None
    if not value.get('cookies', {}).get('hy_token') or not value.get('signature', {}).get('uskey'):
        raise RuntimeError('元宝凭据不完整，请重新导出')
    return value


def headers(value, path, content_type):
    signature = value['signature']
    h = {
        'Cookie': '; '.join(f'{k}={v}' for k, v in value['cookies'].items() if k),
        'Content-Type': content_type,
        'User-Agent': signature['user_agent'],
        'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'X-Instance-ID': '5',
        'X-Uskey': signature['uskey'],
        'X-Bus-Params-Md5': signature['bus_params_md5'],
        'X-Timestamp': signature['timestamp'],
        'X-HY92': signature['hy92'],
        'X-HY93': signature['hy93'],
        'X-device-id': signature['hy93'],
        'X-HY106': '',
        'X-Language': 'zh-CN',
        'X-Requested-With': 'XMLHttpRequest',
        'X-Platform': 'win',
        'X-os_version': 'Windows(10)-Blink',
        'X-Source': 'web',
        'X-Web-Third-Source': 'main',
        'X-WebVersion': '2.87.1',
        'X-ybuitest': '0',
        'X-webdriver': '0',
        'X-Exp-Params': 'enableNewPcStyle=2',
        'x-commit-tag': '8ce400df',
        'X-AgentID': AGENT_ID + (('/' + path.rsplit('/', 1)[-1]) if '/api/chat/' in path else ''),
        'Referer': BASE + '/chat/' + AGENT_ID + (('/' + path.rsplit('/', 1)[-1]) if '/api/chat/' in path else ''),
    }
    if '/api/chat/' in path:
        h.update({'X-Event-Input-Type': '11', 'chat_version': 'v1', 'X-Input-Type': 'text',
                  'X-Traceparent': uuid.uuid4().hex, 'X-Trid-Channel': 'undefined', 'x-web-ch-id': 'null'})
    return h


def post(value, path, body, content_type='application/json'):
    data = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data, headers=headers(value, path, content_type), method='POST')
    try:
        return urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'元宝 HTTP {error.code}: {error.read(300).decode("utf-8", errors="replace")}') from None


def create_chat(value):
    with post(value, '/api/user/agent/conversation/create', {'agentId': AGENT_ID}) as response:
        result = json.load(response)
    if not result.get('id'):
        raise RuntimeError(f'元宝没有返回会话 ID：{str(result)[:200]}')
    return result['id']


def delete_chat(value, conversation_id):
    body = {'conversationIds': [conversation_id], 'uiOptions': {'noToast': True}}
    with post(value, '/api/user/agent/conversation/v1/clear', body) as response:
        text = response.read(500).decode('utf-8', errors='replace')
    if text and text not in ('{}', 'null'):
        try:
            result = json.loads(text)
        except ValueError:
            raise RuntimeError(f'元宝删除响应异常：{text[:200]}') from None
        if isinstance(result, dict) and result.get('code') not in (None, 0):
            raise RuntimeError(f'元宝删除失败：{text[:200]}')


def chat(value, prompt, conversation_id):
    ext_info = {'modelId': 'hunyuan_gpt_175B_0404',
                'agentModeModelSetting': {'modelId': 'hunyuan_gpt_175B_0404'},
                'supportFunctions': {'internetSearch': ''}, 'internetSearch': ''}
    body = {
        'model': 'gpt_175B_0404', 'prompt': prompt, 'plugin': '', 'displayPrompt': prompt,
        'displayPromptType': 1, 'agentId': AGENT_ID, 'isTemporary': False, 'projectId': '',
        'chatModelId': 'hunyuan_gpt_175B_0404', 'supportFunctions': [], 'docOpenid': '',
        'options': {'imageIntention': {'needIntentionModel': True, 'backendUpdateFlag': 2, 'intentionStatus': True}},
        'multimedia': [], 'supportHint': 1,
        'chatModelExtInfo': json.dumps(ext_info, ensure_ascii=False, separators=(',', ':')),
        'applicationIdList': [], 'version': 'v2', 'extReportParams': None,
        'isAtomInput': False, 'conversationId': conversation_id,
        'offsetOfHour': 8, 'offsetOfMinute': 0,
    }
    answer = []
    message_id = None
    with post(value, '/api/chat/' + conversation_id, body, 'text/plain;charset=UTF-8') as response:
        if 'text/event-stream' not in response.headers.get('content-type', ''):
            raise RuntimeError('元宝没有返回聊天流')
        for raw in response:
            line = raw.decode('utf-8', errors='replace').strip()
            if not line.startswith('data:'):
                continue
            try:
                packet = json.loads(line[5:].strip())
            except ValueError:
                continue
            if packet.get('type') == 'error':
                raise RuntimeError(f'元宝聊天失败：{packet.get("msg", "未知错误")}')
            if packet.get('type') == 'text' and packet.get('msg'):
                part = packet['msg']
                answer.append(part)
                print(part, end='', flush=True)
            if packet.get('type') == 'meta':
                message_id = packet.get('messageId')
    if answer:
        print(flush=True)
    if not answer or not message_id:
        raise RuntimeError('元宝聊天流不完整，未保存会话状态')
    return {'conversation_id': conversation_id, 'message_id': message_id}


def read_state():
    try:
        state = json.loads(STATE.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('没有可续聊或删除的元宝会话') from None
    if not state.get('conversation_id'):
        raise RuntimeError('元宝会话状态文件无效')
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
        print(f'已删除元宝会话 {cid}', file=sys.stderr)
        return
    prior = read_state() if args.continuing or (not args.new and not args.delete_after and STATE.exists()) else None
    cid = prior['conversation_id'] if prior else create_chat(value)
    state = None
    try:
        state = chat(value, prompt, cid)
        print(f'会话 ID: {cid}', file=sys.stderr)
        if not args.delete_after:
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    finally:
        if args.delete_after:
            delete_chat(value, cid)
            print(f'已删除本次元宝会话 {cid}', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
