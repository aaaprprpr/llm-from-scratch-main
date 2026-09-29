from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from dataset.label.backend.deepseek_web import QwenWebClient, DeepSeekWebHttpError, WebChatClient


class QwenWebTests(unittest.TestCase):
    def setUp(self):
        # Pacing is verified separately; request-format tests should not wait 30 seconds.
        self.pace = patch.object(QwenWebClient, "_request_gap", 0.0)
        self.next_start = patch.object(QwenWebClient, "_next_start", 0.0)
        self.pace.start()
        self.next_start.start()
        self.addCleanup(self.pace.stop)
        self.addCleanup(self.next_start.stop)
    def test_fixed_session_independent_documents_and_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auth = root / "auth.json"
            auth.write_text(json.dumps({
                "query": "biz_id=ai_qwen&ut=test",
                "cookies": {
                    "tongyi_sso_ticket_hash": "test-ticket",
                    "XSRF-TOKEN": "test-xsrf",
                },
                "storage": {"scene_list": {
                    "qwen_chat": {
                        "eo-clt-actkn": "test",
                        "eo-clt-actkn-dl": 1,
                        "eo-clt-bacsft": [],
                    },
                    "qwen_web": {"eo-clt-dvidn": "device", "eo-clt-snver": "1", "eo-clt-actkn": "base"},
                }},
            }), encoding="utf-8")
            stub = root / "stub.mjs"
            stub.write_text(r"""
globalThis.fetch = async (url, options) => {
  const address = new URL(url);
  if (address.pathname === '/security/external/access/refresh') {
    if (options.headers['eo-clt-actkn'] !== 'base') throw Error('invalid refresh');
    return new Response(JSON.stringify({status: 0, data: {
      'eo-clt-bacsft': ['base-signature'],
      unifyRelate: [{businessScene: 'qwen_chat', 'eo-clt-actkn': 'fresh',
        'eo-clt-actkn-dl': Math.floor(Date.now() / 1000) + 86400,
        'eo-clt-bacsft': Array.from({length: 30}, (_, i) => `signature-${i}`)}],
    }}));
  }
  const body = JSON.parse(options.body);
  if (address.pathname === '/api/v1/session/delete') {
    if (!body.session_id || body.biz_id !== 'ai_qwen' || address.searchParams.get('ut') !== 'test')
      throw Error('invalid deletion');
    return new Response(JSON.stringify({code: 0, success: true}));
  }
  if (address.pathname !== '/api/v2/chat') throw Error('unexpected path');
  if (body.parent_req_id !== '0' || body.scene_param !== 'first_turn' ||
      body.chat_mode !== 'quick' || body.deep_search !== null ||
      !body.session_id || !body.topic_id || !body.messages[0].content.includes('正文'))
    throw Error('invalid independent chat');
  if (!options.headers['clt-acs-sign'] || !options.headers['eo-clt-sacsft'])
    throw Error('missing signature');
  return new Response('data: ' + JSON.stringify({data: {messages: [
    {mime_type: 'multi_load/iframe', content: '{"ok":true}'}
  ]}}) + '\n\n', {headers: {'content-type': 'text/event-stream'}});
};
""", encoding="utf-8")
            state_path = root / "batch_0.json"
            client = QwenWebClient(str(auth), 20, state_path)
            with patch.dict(os.environ, {"NODE_OPTIONS": f"--import={stub}"}):
                first = client.complete([{"role": "user", "content": "第一段正文"}])
                original_state = json.loads(state_path.read_text())
                second = client.complete([{"role": "user", "content": "第二段正文"}])
                second_state = json.loads(state_path.read_text())
                self.assertTrue(client.delete())
            self.assertEqual(first["choices"][0]["message"]["content"], '{"ok":true}')
            self.assertEqual(second["choices"][0]["message"]["content"], '{"ok":true}')
            self.assertEqual(original_state["session_id"], second_state["session_id"])
            self.assertEqual(original_state["topic_id"], second_state["topic_id"])
            self.assertIsNotNone(original_state["topic_id"])
            self.assertFalse(state_path.exists())
            self.assertEqual(len(json.loads(auth.read_text())["storage"]["scene_list"]["qwen_chat"]["eo-clt-bacsft"]), 28)

    def test_application_validation_rejection_disables_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auth = root / "auth.json"
            auth.write_text(json.dumps({
                "query": "biz_id=ai_qwen&ut=test",
                "cookies": {"tongyi_sso_ticket_hash": "ticket"},
                "storage": {"scene_list": {
                    "qwen_chat": {"eo-clt-actkn": "token", "eo-clt-actkn-dl": 4102444800,
                                  "eo-clt-bacsft": ["sign"] * 21},
                    "qwen_web": {"eo-clt-dvidn": "device", "eo-clt-snver": "1"},
                }},
            }), encoding="utf-8")
            stub = root / "stub.mjs"
            stub.write_text("""globalThis.fetch = async () => new Response(
              JSON.stringify({ret: ['FAIL_SYS_USER_VALIDATE', '被挤爆啦']}),
              {headers: {'content-type': 'application/json'}});
            """, encoding="utf-8")
            client = QwenWebClient(str(auth), 20, root / "state.json")
            with patch.dict(os.environ, {"NODE_OPTIONS": f"--import={stub}"}):
                with self.assertRaises(DeepSeekWebHttpError) as raised:
                    client.complete([{"role": "user", "content": "测试"}])
            self.assertEqual(raised.exception.status, 429)
            self.assertIn("应用层拒绝", str(raised.exception))

    def test_batch_slots_share_one_request_start_interval(self):
        started = []

        def fake_run(_client, _prompt, _session_id, _topic_id):
            started.append(time.monotonic())
            return {"choices": []}, "session", None

        clients = [QwenWebClient("auth", 20), QwenWebClient("auth", 20)]
        with (patch.object(WebChatClient, "_run", fake_run),
              patch.object(QwenWebClient, "_next_start", 0.0),
              patch.object(QwenWebClient, "_request_gap", 0.05)):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda client: client._run("test", None, None), clients))
        self.assertEqual(len(started), 2)
        self.assertGreaterEqual(started[1] - started[0], 0.04)

    def test_web_style_json_fence_is_accepted(self):
        from dataset.label.backend.llm_cleaning import LlmCleaner
        assessment = LlmCleaner._parse_assessment(
            '```json\n{"decision":"keep","removals":[],"edits":[],"joins":[],"summary":"保留。"}\n```'
        )
        self.assertEqual(assessment.decision, "keep")

    def test_shared_cleaner_uses_qwen_web_transport_and_fixed_slots(self):
        from dataset.label.backend.llm_cleaning import CleaningConfig, LlmCleaner
        config = CleaningConfig(
            provider="qwen_web", base_url="https://chat2.qianwen.com",
            model="Qwen-web", api_key="/tmp/fake-auth.json",
            context_tokens=32768, max_output_tokens=4096,
        )
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary) / "llm_suggestions"
            with patch("dataset.label.backend.deepseek_web.QwenWebClient.shared") as shared:
                shared.return_value.complete.return_value = {"choices": []}
                for slot in (0, 1, 0):
                    LlmCleaner(config, reports, web_session_key=f"batch_{slot}")._request(
                        "/chat/completions", {"messages": [{"role": "user", "content": "正文"}]},
                    )
            paths = [call.args[2] for call in shared.call_args_list]
            self.assertEqual(paths[0], paths[2])
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual({path.name for path in paths}, {
                "qwen_web_batch_0_session.json", "qwen_web_batch_1_session.json",
            })



if __name__ == "__main__":
    unittest.main()
