from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from dataset.label.backend.deepseek_web import DoubaoWebClient, KimiWebClient
from dataset.label.backend.llm_cleaning import LlmCleaner
from dataset.label.backend.model_settings import ModelSettings

ROOT = Path(__file__).resolve().parents[3]


class AdditionalWebTests(unittest.TestCase):
    def setUp(self):
        for target, name, value in (
            (KimiWebClient, "_request_gap", 0.0),
            (KimiWebClient, "_next_start", 0.0),
        ):
            active = patch.object(target, name, value)
            active.start()
            self.addCleanup(active.stop)

    def test_kimi_structured_request_reuses_one_chat_and_deletes_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auth = root / 'kimi-auth.json'
            auth.write_text(json.dumps({'accessToken': 'fake-access', 'refreshToken': 'fake-refresh'}))
            stub = root / 'stub.mjs'
            stub.write_text(r'''
const id = '01234567-89ab-cdef-0123-456789abcdef';
function frame(value) {
  const data = Buffer.from(JSON.stringify(value));
  const result = Buffer.alloc(data.length + 5);
  result.writeUInt32BE(data.length, 1);
  data.copy(result, 5);
  return result;
}
globalThis.fetch = async (url, options) => {
  const path = new URL(url).pathname;
  if (path.endsWith('/DeleteChat')) {
    const body = JSON.parse(options.body);
    if (body.chat_id !== id) throw Error('wrong chat deleted');
    return new Response(JSON.stringify({chatId: id}));
  }
  if (!path.endsWith('/Chat')) throw Error('unexpected request');
  const body = Buffer.from(options.body);
  const payload = JSON.parse(body.subarray(5).toString());
  if ((payload.chat_id || null) !== (process.env.EXPECT_REUSE ? id : null)) throw Error('chat not reused');
  if (payload.tools.length || payload.options.thinking || payload.options.enable_plugin) throw Error('tools or thinking enabled');
  const text = '{"decision":"keep","removals":[],"edits":[],"joins":[],"summary":"保留。"}';
  return new Response(frame({chat: {id}, op: 'append', block: {text: {content: text}}}),
    {headers: {'content-type': 'application/connect+json'}});
};
''', encoding='utf-8')
            client = KimiWebClient(str(auth), 20, root / 'kimi-state.json')
            with patch.dict(os.environ, {'NODE_OPTIONS': f'--import={stub}'}):
                first = client.complete([{'role': 'user', 'content': '第一条'}])
                with patch.dict(os.environ, {'EXPECT_REUSE': '1'}):
                    second = client.complete([{'role': 'user', 'content': '第二条'}])
                self.assertTrue(client.delete())
            self.assertEqual(first['choices'][0]['message']['content'],
                             second['choices'][0]['message']['content'])
            self.assertFalse((root / 'kimi-state.json').exists())

    def test_doubao_structured_request_carries_conversation_state_and_deletes_it(self):
        spec = importlib.util.spec_from_file_location('doubao_chat_test', ROOT / 'deepseek-web-api/doubao_chat.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []

        class Response(io.BytesIO):
            def __init__(self, content: bytes, content_type: str):
                super().__init__(content)
                self.headers = {'content-type': content_type}
                self.status = 200

        def fake_call(_auth, path, body, *, stream=False):
            calls.append((path, body))
            if path.startswith('/im/'):
                return Response(b'{"status_code":0}', 'application/json')
            self.assertEqual(body['option']['need_deep_think'], 1)
            self.assertEqual(body['ext']['use_deep_think'], '1')
            self.assertFalse(body['option']['connector_info_list'])
            ack = 'event: SSE_ACK\ndata: {"ack_client_meta":{"conversation_id":"12345","section_id":"67"}}\n\n'
            msg = 'event: STREAM_MSG_NOTIFY\ndata: {"meta":{"index_in_conv":2},"content":{"content_block":[{"content":{"text_block":{"text":"{\\"decision\\":\\"keep\\"}"}}}]}}\n\n'
            return Response((ack + msg).encode(), 'text/event-stream')

        def run(mode: str, payload: dict) -> dict:
            with patch.object(module, 'load_auth', return_value={}), patch.object(module, 'call', side_effect=fake_call), \
                 patch.object(sys, 'argv', ['doubao_chat.py', mode]), \
                 patch.object(sys, 'stdin', io.StringIO(json.dumps(payload))), redirect_stdout(io.StringIO()) as out:
                module.main()
                return json.loads(out.getvalue().splitlines()[-1])

        first = run('--json-stdin', {'prompt': '第一条', 'session_id': None, 'topic_id': None})
        second = run('--json-stdin', {'prompt': '第二条', 'session_id': first['session_id'],
                                       'topic_id': first['topic_id']})
        self.assertEqual((first['session_id'], second['session_id']), ('12345', '12345'))
        self.assertTrue(calls[0][1]['option']['need_create_conversation'])
        self.assertFalse(calls[1][1]['option']['need_create_conversation'])
        self.assertEqual(calls[1][1]['client_meta']['last_section_id'], '67')
        self.assertEqual(calls[1][1]['client_meta']['last_message_index'], 2)
        self.assertEqual(run('--delete-session-stdin', {'session_id': '12345'}), {'deleted': '12345'})
        self.assertEqual(calls[-1][0], '/im/conversation/batch_del_user_conv')
        self.assertEqual(DoubaoWebClient('auth', 10)._command_argv('--json-stdin')[0], sys.executable)

    def test_doubao_stream_rate_limit_reports_code_without_verification_data(self):
        spec = importlib.util.spec_from_file_location('doubao_rate_test', ROOT / 'deepseek-web-api/doubao_chat.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Response(io.BytesIO):
            headers = {'content-type': 'text/event-stream'}
            status = 200

        packet = {'error_code': 710022004, 'error_msg': 'rate limited',
                  'extra': {'decision': 'verification-secret'}}
        stream = ('event: STREAM_ERROR\ndata: ' + json.dumps(packet) + '\n\n').encode()
        with patch.object(module, 'call', return_value=Response(stream)):
            with self.assertRaises(RuntimeError) as raised:
                module.chat({}, '测试', None, structured=True)
        self.assertIn('rate limited', str(raised.exception))
        self.assertIn('710022004', str(raised.exception))
        self.assertNotIn('verification-secret', str(raised.exception))

    def test_web_client_saves_updated_conversation_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / 'doubao-state.json'
            client = DoubaoWebClient('fake-auth', 10, state)
            cursors = []

            def fake_run(_prompt, session_id, topic_id):
                cursors.append((session_id, topic_id))
                next_cursor = f'cursor-{len(cursors)}'
                return {'choices': [{'message': {'content': 'ok'}}]}, '12345', next_cursor

            with patch.object(client, '_run', side_effect=fake_run):
                for _ in range(3):
                    client.complete([{'role': 'user', 'content': 'doc'}])
            self.assertEqual(cursors, [(None, None), ('12345', 'cursor-1'),
                                       ('12345', 'cursor-2')])
            self.assertEqual(json.loads(state.read_text())['topic_id'], 'cursor-3')

    def test_new_sources_share_cleaner_and_model_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            web = root / 'deepseek-web-api'
            web.mkdir()
            sources = {
                'kimi_web': '.kimi-web-auth.json',
                'doubao_web': '.doubao-web-auth.json',
                'chatglm_web': '.chatglm-web-auth.json',
                'spark_web': '.spark-web-auth.json',
                'wenxin_web': '.wenxin-web-auth.json',
                'yuanbao_web': '.yuanbao-web-auth.json',
            }
            for name in sources.values():
                (web / name).write_text('{}')
            settings = ModelSettings(root)
            settings.project_root = root
            for source in sources:
                config = settings.config(source)
                self.assertEqual(config.provider, source)
                self.assertTrue(config.api_key)
                cleaner = LlmCleaner(config, root / 'llm_suggestions', web_session_key='batch_0')
                with patch('dataset.label.backend.deepseek_web.WEB_CLIENTS') as clients:
                    clients.__contains__.return_value = True
                    clients.__getitem__.return_value.shared.return_value.complete.return_value = {'choices': []}
                    self.assertEqual(cleaner._request('/chat/completions', {'messages': []}), {'choices': []})
                    self.assertIn(f'{source}_batch_0_session.json',
                                  str(clients.__getitem__.return_value.shared.call_args.args[2]))
