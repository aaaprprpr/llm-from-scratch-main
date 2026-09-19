from __future__ import annotations

import http.client
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from label.backend.llm_cleaning import CleaningConfig, LlmCleaner, LlmCleaningError


API_KEY = "sk-unit-test-do-not-persist"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def completion(**changes):
    assessment = {
        "decision": "keep", "quality": 2, "category": "encyclopedia",
        "removals": [], "edits": [], "joins": [], "summary": "保留正文，删除广告。",
    }
    assessment.update(changes)
    return {
        "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(assessment, ensure_ascii=False),
        }}],
        "usage": {"prompt_tokens": 125, "completion_tokens": 35},
    }


class RemoteLlmCleaningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.reports = self.directory / "reports"
        self.config_path = self.directory / "label.json"
        self.env_path = self.directory / ".env"
        self.requests = []

    def config(self, **changes):
        settings = {
            "provider": "dashscope", "api_key": API_KEY,
            "base_url": BASE_URL, "model": "qwen-plus", "context_tokens": 32768,
        }
        settings.update(changes)
        return CleaningConfig(**settings)

    def write_config(self, **settings):
        self.config_path.write_text(json.dumps({"llm_cleaning": settings}), encoding="utf-8")

    def transport(self, responses):
        pending = iter(responses)

        def open_request(request, timeout):
            self.requests.append(request)
            self.assertEqual(request.full_url, BASE_URL + "/chat/completions")
            self.assertEqual(request.get_method(), "POST")
            self.assertGreater(timeout, 0)
            response = next(pending, responses[-1])
            if isinstance(response, Exception):
                raise response
            return io.BytesIO(json.dumps(response, ensure_ascii=False).encode("utf-8"))

        return open_request

    def clean(self, responses, blocks=None, config=None):
        cleaner = LlmCleaner(config or self.config(), self.reports)
        with patch.object(cleaner._opener, "open", side_effect=self.transport(responses)):
            return cleaner.clean(
                blocks or [{"id": "body", "text": "原始百科正文。", "separator_after": ""}],
                title="百科条目", provenance={"dataset": "wiki_zh_20231101", "doc_id": "source-1"},
            )

    def assert_no_reports(self):
        self.assertEqual(list(self.reports.glob("*")), [])

    def test_remote_cleaning_uses_completion_endpoint_and_preserves_report_contract(self):
        blocks = [{"id": "body", "text": "原始百科正文。点击领取优惠！后续正文。"}]
        result = self.clean([completion(removals=[{"unit_id": 1, "reason": "advertisement"}])], blocks)
        self.assertEqual(result["edited_text"], "原始百科正文。后续正文。")
        self.assertEqual(result["removals"][0]["text"], "点击领取优惠！")
        self.assertEqual(result["input_tokens"], 125)
        self.assertEqual(result["output_tokens"], 35)
        self.assertEqual(len(self.requests), 1)

        request = self.requests[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer " + API_KEY)
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen-plus")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIs(payload["enable_thinking"], False)
        self.assertIs(payload["stream"], False)
        self.assertNotIn("chat_template_kwargs", payload)
        system_prompt = payload["messages"][0]["content"]
        self.assertIn('"properties"', system_prompt)
        self.assertIn('"required"', system_prompt)

        reports = list(self.reports.glob("*.json"))
        self.assertEqual(len(reports), 1)
        report_text = reports[0].read_text(encoding="utf-8")
        self.assertNotIn(API_KEY, report_text)
        self.assertNotIn("Authorization", report_text)
        self.assertNotIn(API_KEY, repr(self.config()))
        report = json.loads(report_text)
        self.assertEqual(report["input_blocks"], blocks)
        self.assertEqual(report["provenance"]["dataset"], "wiki_zh_20231101")
        self.assertEqual(report["result"], result)
        self.assertEqual(report["context_tokens"], 32768)

    def test_remote_chunks_keep_all_original_text_without_local_tokenizer_calls(self):
        blocks = [{"id": str(index), "text": f"第{index}段正文。", "separator_after": "\n\n"}
                  for index in range(30)]
        result = self.clean([completion()], blocks)
        self.assertGreater(result["chunks"], 1)
        self.assertEqual(len(self.requests), result["chunks"])
        self.assertEqual(result["edited_text"], "".join(b["text"] + b["separator_after"] for b in blocks))
        self.assertFalse(result["text_changed"])

    def test_remote_invalid_suggestion_retries_with_feedback_before_saving(self):
        bad = completion(edits=[{"unit_id": 0, "replacement": "凭空新增文字"}])
        result = self.clean([bad, completion()])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(len(self.requests), 2)
        retry_payload = json.loads(self.requests[1].data)
        retry_input = json.loads(retry_payload["messages"][1]["content"])
        self.assertIn("retry_instruction", retry_input)
        self.assertEqual(result["edited_text"], "原始百科正文。")
        self.assertEqual(result["input_tokens"], 250)
        report_text = next(self.reports.glob("*.json")).read_text(encoding="utf-8")
        self.assertNotIn(API_KEY, report_text)

    def test_remote_still_rejects_invalid_schema_and_unsafe_source_edits(self):
        for response in [
            completion(quality="2"),
            completion(category="unsupported"),
            completion(unexpected="not allowed"),
            completion(removals=[{"unit_id": 9, "reason": "advertisement"}]),
            completion(removals=[{"unit_id": 0, "reason": "advertisement"}]),
            completion(decision="drop"),
            completion(edits=[{"unit_id": 0, "replacement": "不存在于原文的观点。"}]),
            completion(joins=[[0, 0]]),
            {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]},
        ]:
            with self.subTest(response=response):
                self.requests.clear()
                with self.assertRaises(LlmCleaningError):
                    self.clean([response])
                self.assertEqual(len(self.requests), 2)
                self.assert_no_reports()

    def test_failure_after_successful_chunk_does_not_write_partial_report(self):
        blocks = [{"id": str(index), "text": "原始正文。"} for index in range(25)]
        with self.assertRaises(LlmCleaningError):
            self.clean([completion(), completion(removals=[{"unit_id": 99, "reason": "markup"}])], blocks)
        self.assertEqual(len(self.requests), 3)
        self.assert_no_reports()

    def test_missing_key_is_actionable_and_never_sends_a_request(self):
        with patch("urllib.request.OpenerDirector.open") as open_request:
            with self.assertRaises(LlmCleaningError) as caught:
                cleaner = LlmCleaner(self.config(api_key=""), self.reports)
                cleaner.clean([{"id": "body", "text": "正文。"}], title=None, provenance={})
        self.assertIn("DASHSCOPE_API_KEY", str(caught.exception))
        open_request.assert_not_called()
        self.assert_no_reports()

    def test_remote_http_errors_are_clear_and_do_not_expose_credentials(self):
        for status in [401, 403, 429, 500]:
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    BASE_URL + "/chat/completions", status, "provider diagnostic " + API_KEY,
                    {}, io.BytesIO(json.dumps({"error": API_KEY}).encode()),
                )
                with self.assertRaises(LlmCleaningError) as caught:
                    self.clean([error])
                message = str(caught.exception)
                self.assertIn(str(status), message)
                self.assertNotIn(API_KEY, message)
                self.assert_no_reports()

    def test_remote_connection_errors_do_not_expose_transport_details(self):
        for error in [urllib.error.URLError("connection failed " + API_KEY), TimeoutError(API_KEY)]:
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(LlmCleaningError) as caught:
                    self.clean([error])
                self.assertNotIn(API_KEY, str(caught.exception))
                self.assert_no_reports()

    def test_incomplete_response_has_clear_error_and_does_not_write_report(self):
        cleaner = LlmCleaner(self.config(), self.reports)
        response = io.BytesIO()
        error = http.client.IncompleteRead(b'{"choices":', 100)
        with patch.object(response, "read", side_effect=error), \
                patch.object(cleaner._opener, "open", return_value=response) as open_request:
            with self.assertRaises(LlmCleaningError) as caught:
                cleaner.clean([{"id": "body", "text": "完整原文。"}], title=None, provenance={})
        self.assertIn("连接", str(caught.exception))
        self.assertIn("第 1/1 块", str(caught.exception))
        self.assertNotIn(API_KEY, str(caught.exception))
        open_request.assert_called_once()
        self.assert_no_reports()

    def test_dotenv_populates_remote_settings_without_mutating_process_environment(self):
        self.write_config(provider="dashscope", base_url=BASE_URL, model="json-model", context_tokens=32768)
        self.env_path.write_text(
            '# Provider settings\nDASHSCOPE_API_KEY="sk-from-env-file"\n'
            "DEFAULT_MODEL='qwen-file-model'\nDASHSCOPE_BASE_URL=https://file.example.test/v1\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {}, clear=True):
            config = CleaningConfig.from_file(self.config_path, self.env_path)
            self.assertEqual(config.api_key, "sk-from-env-file")
            self.assertEqual(config.model, "qwen-file-model")
            self.assertEqual(config.base_url, "https://file.example.test/v1")
            self.assertNotIn("DASHSCOPE_API_KEY", os.environ)
            self.assertNotIn("DEFAULT_MODEL", os.environ)

    def test_process_environment_takes_precedence_over_dotenv_and_json(self):
        self.write_config(provider="dashscope", base_url=BASE_URL, model="json-model", context_tokens=32768)
        self.env_path.write_text(
            "DASHSCOPE_API_KEY=sk-file\nDEFAULT_MODEL=file-model\n"
            "DASHSCOPE_BASE_URL=https://file.example.test/v1\n", encoding="utf-8",
        )
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "sk-process", "DEFAULT_MODEL": "process-model",
            "DASHSCOPE_BASE_URL": "https://process.example.test/v1",
        }, clear=True):
            config = CleaningConfig.from_file(self.config_path, self.env_path)
        self.assertEqual(config.api_key, "sk-process")
        self.assertEqual(config.model, "process-model")
        self.assertEqual(config.base_url, "https://process.example.test/v1")

    def test_missing_optional_env_file_keeps_json_remote_settings(self):
        self.write_config(provider="dashscope", base_url=BASE_URL, model="configured-model", context_tokens=32768)
        with patch.dict(os.environ, {}, clear=True):
            config = CleaningConfig.from_file(self.config_path, self.env_path)
        self.assertEqual(config.provider, "dashscope")
        self.assertEqual(config.base_url, BASE_URL)
        self.assertEqual(config.model, "configured-model")
        self.assertEqual(config.api_key, "")

    def test_empty_process_key_does_not_fall_back_to_dotenv_secret(self):
        self.write_config(provider="dashscope", base_url=BASE_URL, model="qwen-plus", context_tokens=32768)
        self.env_path.write_text("DASHSCOPE_API_KEY=sk-from-file\n", encoding="utf-8")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}, clear=True):
            config = CleaningConfig.from_file(self.config_path, self.env_path)
        self.assertEqual(config.api_key, "")

    def test_remote_url_requires_https_and_rejects_embedded_credentials(self):
        for url in [
            "http://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://example.test/v1?api_key=" + API_KEY,
            "https://user:" + API_KEY + "@example.test/v1",
            "https://example.test/v1#" + API_KEY,
            "https://example.test/compatible-mode",
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError) as caught:
                self.config(base_url=url)
            self.assertNotIn(API_KEY, str(caught.exception))

    def test_legacy_local_config_ignores_remote_environment(self):
        self.write_config(model="local-model")
        self.env_path.write_text("DASHSCOPE_API_KEY=sk-file\nDEFAULT_MODEL=file-model\n", encoding="utf-8")
        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": API_KEY, "DEFAULT_MODEL": "remote-model",
            "DASHSCOPE_BASE_URL": BASE_URL,
        }, clear=True):
            config = CleaningConfig.from_file(self.config_path, self.env_path)
        self.assertEqual(config.provider, "llamacpp")
        self.assertEqual(config.base_url, "http://127.0.0.1:8080")
        self.assertEqual(config.model, "local-model")
        self.assertEqual(config.api_key, "")


if __name__ == "__main__":
    unittest.main()
