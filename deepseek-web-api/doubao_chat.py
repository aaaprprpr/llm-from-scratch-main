"""豆包网页登录态直连聊天；Python 3 标准库，无运行时浏览器依赖。"""

import argparse
import copy
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def env_value(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in (ROOT / '.env').read_text(encoding='utf-8').splitlines():
            if line.lstrip().startswith(name + '='):
                value = line.split('=', 1)[1].strip()
                if value.startswith('"'):
                    return json.loads(value)
                return value.strip("'")
    except FileNotFoundError:
        pass
    return None


AUTH_PATH = ROOT / (env_value('DOUBAO_WEB_AUTH_FILE') or '.doubao-web-auth.json')
STATE_PATH = ROOT / '.doubao-session.json'
TEMPLATE_PATH = ROOT / 'doubao_request_template.json'


def load_auth():
    try:
        auth = json.loads(AUTH_PATH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('缺少豆包凭据；先运行 python doubao_capture_auth.py <调试端口>') from None
    if not auth.get('cookies') or not auth.get('query'):
        raise RuntimeError('豆包凭据文件无效，请重新导出')
    return auth


def load_state():
    try:
        state = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('没有可继续或删除的豆包会话') from None
    if not str(state.get('conversation_id', '')).isdigit():
        raise RuntimeError('豆包会话状态文件无效')
    return state


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def clear_state(conversation_id):
    if STATE_PATH.exists() and load_state()['conversation_id'] == conversation_id:
        STATE_PATH.unlink()


def call(auth, path, body, *, stream=False):
    url = f'https://www.doubao.com{path}?{auth["query"]}'
    headers = {
        'Content-Type': 'application/json; encoding=utf-8' if path.startswith('/im/') else 'application/json',
        'Cookie': '; '.join(f'{key}={value}' for key, value in auth['cookies'].items()),
        'Origin': 'https://www.doubao.com',
        'Referer': 'https://www.doubao.com/',
    }
    if path.startswith('/im/'):
        headers['Agw-Js-Conv'] = 'str'
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8'), headers=headers, method='POST')
    try:
        return urllib.request.urlopen(request, timeout=300 if stream else 60)
    except urllib.error.HTTPError as error:
        detail = error.read(300).decode('utf-8', errors='replace')
        raise RuntimeError(f'豆包 HTTP {error.code}: {detail}') from None


def build_body(prompt, state):
    body = json.loads(TEMPLATE_PATH.read_text(encoding='utf-8'))
    now = int(time.time() * 1000)
    body['client_meta']['local_conversation_id'] = state.get('local_conversation_id') if state else f'local_{now}'
    body['client_meta']['conversation_id'] = state['conversation_id'] if state else ''
    body['client_meta']['last_section_id'] = state.get('section_id', '') if state else ''
    body['client_meta']['last_message_index'] = state.get('last_message_index') if state else None
    body['messages'][0]['local_message_id'] = str(uuid.uuid4())
    block = body['messages'][0]['content_block'][0]
    block['block_id'] = str(uuid.uuid4())
    block['content']['text_block']['text'] = prompt
    body['option']['create_time_ms'] = now
    body['option']['unique_key'] = str(uuid.uuid4())
    body['option']['need_create_conversation'] = state is None
    body['option']['recovery_option']['req_create_time_sec'] = now // 1000
    body['option']['need_deep_think'] = 1
    body['option']['model_config']['model_item_key'] = '0'  # 沿用抓包的模型键；思考模式由下方开关控制。
    body['option']['connector_info_list'] = []
    body['ext']['use_deep_think'] = '1'
    return body


def chat(auth, prompt, state, *, structured=False, on_created=None):
    body = build_body(prompt, state)
    result = copy.deepcopy(state) if state else {'conversation_id': '', 'local_conversation_id': body['client_meta']['local_conversation_id']}
    answer = []
    event = None
    data_lines = []
    event_counts = {}

    def handle():
        nonlocal event, data_lines
        if event:
            event_counts[event] = event_counts.get(event, 0) + 1
        if not data_lines:
            event = None
            return
        try:
            value = json.loads('\n'.join(data_lines))
        except json.JSONDecodeError:
            event, data_lines = None, []
            return
        if event == 'SSE_ACK':
            meta = value.get('ack_client_meta', {})
            conversation_id = str(meta.get('conversation_id') or result['conversation_id'])
            if not result['conversation_id'] and conversation_id and on_created:
                on_created(conversation_id)
            result['conversation_id'] = conversation_id
            result['section_id'] = str(meta.get('section_id') or result.get('section_id', ''))
        elif event == 'STREAM_MSG_NOTIFY':
            meta = value.get('meta', {})
            if meta.get('index_in_conv') is not None:
                result['last_message_index'] = meta['index_in_conv']
            for block in value.get('content', {}).get('content_block', []):
                part = block.get('content', {}).get('text_block', {}).get('text', '')
                if part:
                    if not structured:
                        print(part, end='', flush=True)
                    answer.append(part)
        elif event == 'STREAM_CHUNK':
            for patch in value.get('patch_op', []):
                if patch.get('patch_object') != 1:
                    continue
                for block in patch.get('patch_value', {}).get('content_block', []):
                    part = block.get('content', {}).get('text_block', {}).get('text', '')
                    if part:
                        if not structured:
                            print(part, end='', flush=True)
                        answer.append(part)
        elif event == 'CHUNK_DELTA':
            part = value.get('text', '')
            if part:
                if not structured:
                    print(part, end='', flush=True)
                answer.append(part)
        elif event in ('SSE_ERROR', 'STREAM_ERROR', 'ERROR'):
            # The extra field can contain verification material; report only
            # the stable code and message needed to stop this source.
            raise RuntimeError(
                f'豆包回复错误：code={value.get("error_code")}，'
                f'message={str(value.get("error_msg", ""))[:120]}'
            )
        event, data_lines = None, []

    with call(auth, '/chat/completion', body, stream=True) as response:
        content_type = response.headers.get('content-type', '')
        if 'text/event-stream' not in content_type:
            empty = response.read(1) == b''
            raise RuntimeError(
                f'豆包没有返回 SSE 聊天流：HTTP {response.status}，Content-Type {content_type or "(空)"}，'
                f'响应{ "为空" if empty else "非空" }'
            )
        for raw in response:
            line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
            if not line:
                handle()
            elif line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                data_lines.append(line[5:].lstrip())
        handle()
    if answer and not structured:
        print(flush=True)
    if not result['conversation_id'] or not answer:
        raise RuntimeError(
            f'豆包响应缺少会话 ID 或回复正文：会话ID缺失={not bool(result["conversation_id"])}，'
            f'正文缺失={not bool(answer)}，SSE事件={event_counts}'
        )
    return result, ''.join(answer)


def delete_chat(auth, conversation_id):
    body = {
        'cmd': 4171,
        'uplink_body': {'batch_delete_user_conversation_uplink_body': {'conversation_id': [conversation_id], 'delete_all': False, 'conversation_type': 3}},
        'sequence_id': str(uuid.uuid4()), 'channel': 2, 'version': '1',
    }
    with call(auth, '/im/conversation/batch_del_user_conv', body) as response:
        value = json.load(response)
        if response.status != 200 or value.get('status_code', 0) != 0:
            raise RuntimeError(f'豆包删除失败：{value.get("status_desc", "未知错误")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--new', action='store_true', help='创建新会话')
    group.add_argument('--continue', dest='continuing', action='store_true', help='明确续聊')
    group.add_argument('--json-stdin', action='store_true', help='从标准输入读取结构化请求')
    group.add_argument('--delete-session-stdin', action='store_true', help='删除指定会话')
    parser.add_argument('--delete-after', action='store_true', help='本次回复后删除本次会话')
    parser.add_argument('--delete-current', action='store_true', help='删除本地当前会话')
    parser.add_argument('prompt', nargs='*')
    args = parser.parse_args()
    prompt = ' '.join(args.prompt).strip()
    if ((not args.delete_current and not args.json_stdin and not args.delete_session_stdin and not prompt)
            or (args.delete_current and (prompt or args.new or args.continuing or args.delete_after))
            or ((args.json_stdin or args.delete_session_stdin) and (prompt or args.delete_after or args.delete_current))):
        parser.error('请提供消息，或单独使用 --delete-current')
    auth = load_auth()
    if args.delete_session_stdin:
        conversation_id = str(json.load(sys.stdin)['session_id'])
        if not conversation_id.isdigit():
            raise RuntimeError('豆包会话 ID 无效')
        delete_chat(auth, conversation_id)
        print(json.dumps({'deleted': conversation_id}))
        return
    if args.delete_current:
        state = load_state()
        delete_chat(auth, state['conversation_id'])
        clear_state(state['conversation_id'])
        print(f'已删除豆包会话 {state["conversation_id"]}', file=sys.stderr)
        return
    if args.json_stdin:
        request = json.load(sys.stdin)
        prompt = request.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError('缺少提示词')
        session_id = request.get('session_id')
        topic_id = request.get('topic_id')
        if session_id is not None:
            if not str(session_id).isdigit() or not isinstance(topic_id, str):
                raise RuntimeError('豆包会话状态无效')
            prior = json.loads(topic_id)
            if prior.get('conversation_id') != session_id:
                raise RuntimeError('豆包会话状态不匹配')
        else:
            prior = None
        result, content = chat(auth, prompt, prior, structured=True,
                               on_created=lambda conversation_id: print(json.dumps({'session_id': conversation_id}), flush=True))
        print(json.dumps({'session_id': result['conversation_id'],
                          'topic_id': json.dumps(result, ensure_ascii=False), 'content': content}, ensure_ascii=False))
        return
    prior = load_state() if args.continuing or (not args.new and not args.delete_after and STATE_PATH.exists()) else None
    created = None
    try:
        created, _ = chat(auth, prompt, prior)
        print(f'会话 ID: {created["conversation_id"]}', file=sys.stderr)
        if not args.delete_after:
            save_state(created)
    finally:
        if args.delete_after and created:
            delete_chat(auth, created['conversation_id'])
            clear_state(created['conversation_id'])
            print(f'已删除本次豆包会话 {created["conversation_id"]}', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
