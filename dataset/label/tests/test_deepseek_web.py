from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset.label.backend.deepseek_web import (
    DeepSeekWebClient, DeepSeekWebHttpError, _node_executable,
)
from dataset.label.backend.llm_cleaning import CleaningConfig, LlmCleaner, LlmCleaningHttpError


class DeepSeekWebTests(unittest.TestCase):
    def setUp(self):
        for target, name, value in (
            (DeepSeekWebClient, "_request_gap", 0.0),
            (DeepSeekWebClient, "_next_start", 0.0),
        ):
            active = patch.object(target, name, value)
            active.start()
            self.addCleanup(active.stop)

    def test_supplied_node_example_disables_thinking_and_search(self):
        script = Path(__file__).resolve().parents[3] / "deepseek-web-api" / "web_chat_once.mjs"
        pow_module = script.with_name("pow23.mjs").as_uri()
        with tempfile.TemporaryDirectory() as temporary:
            stub = Path(temporary) / "stub.mjs"
            stub.write_text(r"""
const { deepseekHash } = await import(__POW_MODULE__);
globalThis.fetch = async (url, options) => {
  const path = new URL(url).pathname;
  const body = JSON.parse(options.body);
  if (options.headers.Authorization !== 'Bearer fake-token') throw Error('missing token');
  if (path.endsWith('/chat_session/create')) {
    if (process.env.EXPECT_REUSE === '1') throw Error('unexpected new session');
    const data = {code: 0, data: {biz_code: 0, biz_data: {id: 'session'}}};
    return new Response(JSON.stringify(data));
  }
  if (path.endsWith('/create_pow_challenge')) {
    const challenge = {
      algorithm: 'DeepSeekHashV1',
      challenge: deepseekHash('salt_expiry_0'),
      salt: 'salt', expire_at: 'expiry', difficulty: 1, signature: 'signature'
    };
    const data = {code: 0, data: {biz_code: 0, biz_data: {challenge}}};
    return new Response(JSON.stringify(data));
  }
  if (!path.endsWith('/chat/completion')) throw Error('unknown path');
  if (body.thinking_enabled !== false || body.search_enabled !== false)
    throw Error('thinking or search enabled');
  if (body.model_type !== 'default' || !body.prompt.includes('正文') ||
      body.chat_session_id !== 'session' || body.parent_message_id !== null)
    throw Error('incorrect request body');
  if (!options.headers['X-DS-PoW-Response']) throw Error('missing proof');
  const packet = {v: {response: {fragments: [
    {type: 'RESPONSE', content: '{"ok":true}'}
  ]}}};
  return new Response('event: ready\ndata: {"response_message_id":2}\n\ndata: ' + JSON.stringify(packet) + '\n\n',
    {headers: {'content-type': 'text/event-stream'}});
};
""".replace("__POW_MODULE__", json.dumps(pow_module)), encoding="utf-8")
            result = subprocess.run(
                [_node_executable(), "--import", str(stub), str(script), "--json-stdin"],
                input=json.dumps({"prompt": "测试正文", "session_id": None}), text=True, capture_output=True,
                env={**os.environ, "DEEPSEEK_WEB_TOKEN": "fake-token"}, timeout=15,
            )
            structured = subprocess.run(
                [_node_executable(), "--import", str(stub), str(script), "--json-stdin"],
                input=json.dumps({"prompt": "第二段正文", "session_id": "session",
                                  "parent_message_id": None}, ensure_ascii=False),
                text=True, capture_output=True,
                env={**os.environ, "DEEPSEEK_WEB_TOKEN": "fake-token", "EXPECT_REUSE": "1"},
                timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["content"], '{"ok":true}')
        self.assertEqual(structured.returncode, 0, structured.stderr)
        self.assertEqual(json.loads(structured.stdout.splitlines()[-1])["session_id"], "session")
        self.assertEqual(json.loads(structured.stdout.splitlines()[-1])["content"], '{"ok":true}')

    def test_client_reuses_persisted_session_with_independent_documents(self):
        replies = [subprocess.CompletedProcess([], 0, json.dumps({
            "content": '{"ok":true}', "session_id": "session-1", "response_message_id": 2,
        }), "")] * 2
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "web_session.json"
            first = DeepSeekWebClient("fake-token", 60, state_path)
            with patch("dataset.label.backend.deepseek_web.subprocess.run", side_effect=replies) as run:
                first.complete([{"role": "system", "content": "rules"},
                                {"role": "user", "content": "first source"}])
                second = DeepSeekWebClient("fake-token", 60, state_path)
                result = second.complete([{"role": "user", "content": "second source"}])
            first_request = json.loads(run.call_args_list[0].kwargs["input"])
            second_request = json.loads(run.call_args_list[1].kwargs["input"])
            self.assertIsNone(first_request["session_id"])
            self.assertEqual(second_request["session_id"], "session-1")
            self.assertIsNone(second_request["parent_message_id"])
            self.assertEqual(result["choices"][0]["message"]["content"], '{"ok":true}')
            self.assertNotIn("fake-token", state_path.read_text())
            self.assertNotIn("second source", " ".join(run.call_args.args[0]))
            self.assertEqual(run.call_args.kwargs["env"]["DEEPSEEK_WEB_TOKEN"], "fake-token")

    def test_first_failed_request_still_keeps_created_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "web_session.json"
            client = DeepSeekWebClient("fake-token", 60, state_path)
            replies = [
                subprocess.CompletedProcess([], 1, '{"session_id":"session-1"}\n',
                                            'completion: HTTP 401'),
                subprocess.CompletedProcess([], 0, json.dumps({
                    "content": "OK", "session_id": "session-1", "response_message_id": 4,
                }), ""),
            ]
            with patch("dataset.label.backend.deepseek_web.subprocess.run", side_effect=replies) as run:
                with self.assertRaises(DeepSeekWebHttpError):
                    client.complete([{"role": "user", "content": "first source"}])
                client.complete([{"role": "user", "content": "second source"}])
            self.assertEqual(json.loads(state_path.read_text())["session_id"], "session-1")
            self.assertEqual(json.loads(run.call_args_list[1].kwargs["input"])["session_id"], "session-1")

    def test_rate_hint_retries_after_cooldown_in_same_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "web_session.json"
            client = DeepSeekWebClient("fake-token", 60, path)
            replies = [
                subprocess.CompletedProcess([], 1, '{"session_id":"session-1"}\n',
                                            'rate_limit_reached: Messages too frequent'),
                subprocess.CompletedProcess([], 0, json.dumps({
                    "content": "OK", "session_id": "session-1", "response_message_id": 4,
                }), ""),
            ]
            with (patch("dataset.label.backend.deepseek_web.subprocess.run", side_effect=replies) as run,
                  patch.object(client, "_wait_turn") as wait_turn,
                  patch.object(client, "_defer") as defer):
                result = client.complete([{"role": "user", "content": "source"}])
            self.assertEqual(result["choices"][0]["message"]["content"], "OK")
            self.assertEqual(wait_turn.call_count, 2)
            defer.assert_called_once_with(15)
            self.assertEqual(json.loads(run.call_args_list[1].kwargs["input"])["session_id"], "session-1")

    def test_invalid_saved_session_is_recreated_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "web_session.json"
            state_path.write_text(json.dumps({
                "token_sha256": hashlib.sha256(b"fake-token").hexdigest(),
                "session_id": "stale",
            }))
            client = DeepSeekWebClient("fake-token", 60, state_path)
            replies = [
                subprocess.CompletedProcess([], 1, "",
                    'completion: HTTP 200, {"data":{"biz_msg":"invalid chat session id"}}'),
                subprocess.CompletedProcess([], 0, json.dumps({
                    "content": "OK", "session_id": "new-session", "response_message_id": 4,
                }), ""),
            ]
            with patch("dataset.label.backend.deepseek_web.subprocess.run", side_effect=replies) as run:
                result = client.complete([{"role": "user", "content": "source"}])
            self.assertEqual(result["choices"][0]["message"]["content"], "OK")
            self.assertEqual(json.loads(run.call_args_list[0].kwargs["input"])["session_id"], "stale")
            self.assertIsNone(json.loads(run.call_args_list[1].kwargs["input"])["session_id"])
            self.assertEqual(json.loads(state_path.read_text())["session_id"], "new-session")

    def test_shared_client_is_one_session_for_single_and_batch_cleaners(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "web_session.json"
            first = DeepSeekWebClient.shared("fake-token", 60, path)
            second = DeepSeekWebClient.shared("fake-token", 60, path)
            self.assertIs(first, second)

    def test_parallel_chunks_do_not_overlap_on_one_conversation(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = DeepSeekWebClient("fake-token", 60, Path(temporary) / "session.json")
            active = peak = 0
            lock = threading.Lock()
            def run(_prompt, session_id, _topic_id):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return {"choices": [{"message": {"content": "ok"}}]}, session_id or "session", None
            with patch.object(client, "_run", side_effect=run):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    results = list(pool.map(lambda n: client.complete([{"role": "user", "content": str(n)}]), range(3)))
            self.assertEqual(len(results), 3)
            self.assertEqual(peak, 1)

    def test_batch_web_slots_use_distinct_stable_session_files(self):
        config = CleaningConfig(
            provider="deepseek_web", base_url="https://chat.deepseek.com",
            model="deepseek-web-default", api_key="fake-token",
            context_tokens=32768, max_output_tokens=4096,
        )
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary) / "llm_suggestions"
            with patch("dataset.label.backend.deepseek_web.DeepSeekWebClient.shared") as shared:
                shared.return_value.complete.return_value = {"choices": []}
                for slot in (0, 1, 0):
                    cleaner = LlmCleaner(config, reports, web_session_key=f"batch_{slot}")
                    cleaner._request("/chat/completions", {"messages": []})
            paths = [call.args[2] for call in shared.call_args_list]
            self.assertEqual(paths[0], paths[2])
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual({path.name for path in paths}, {
                "deepseek_web_batch_0_session.json", "deepseek_web_batch_1_session.json",
            })

    def test_web_http_status_reaches_shared_cleaner(self):
        config = CleaningConfig(
            provider="deepseek_web", base_url="https://chat.deepseek.com",
            model="deepseek-web-default", api_key="fake-token",
            context_tokens=32768, max_output_tokens=4096,
        )
        cleaner = LlmCleaner(config, Path("."))
        with patch("dataset.label.backend.deepseek_web.subprocess.run", return_value=subprocess.CompletedProcess(
            [], 1, "", "HTTP 401: fake-token expired",
        )):
            with self.assertRaises(LlmCleaningHttpError) as caught:
                cleaner._request("/chat/completions", {"messages": []})
        self.assertEqual(caught.exception.status, 401)
        self.assertNotIn("fake-token", str(caught.exception))

    def test_shared_cleaner_accepts_web_response(self):
        config = CleaningConfig(
            provider="deepseek_web", base_url="https://chat.deepseek.com",
            model="deepseek-web-default", api_key="fake-token",
            context_tokens=32768, max_output_tokens=4096,
        )
        assessment = {
            "decision": "keep",
            "removals": [], "edits": [], "joins": [], "summary": "保留正文。",
        }
        with tempfile.TemporaryDirectory() as temporary:
            cleaner = LlmCleaner(config, Path(temporary))
            with patch("dataset.label.backend.deepseek_web.DeepSeekWebClient.complete", return_value={
                "choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(assessment, ensure_ascii=False),
                }}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }) as complete:
                result = cleaner.clean(
                    [{"id": "body", "text": "这是一段可读正文。"}],
                    title="样例", provenance={"doc_id": "sample"},
                )
            self.assertIn("这是一段可读正文。", result["edited_text"])
            self.assertEqual(complete.call_count, 1)
            self.assertEqual(complete.call_args.args[0][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
