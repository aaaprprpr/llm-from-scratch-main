from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from label.backend.llm_cleaning import CleaningConfig, LlmCleaner, LlmCleaningError, split_units


def completion(removals=(), decision="keep", finish_reason="stop", edits=(), joins=()):
    assessment = {
        "decision": decision, "quality": 2, "category": "encyclopedia",
        "removals": [{"unit_id": index, "reason": "advertisement"} for index in removals],
        "edits": list(edits), "joins": list(joins),
        "summary": "保留正文，删除广告。",
    }
    return {"choices": [{"finish_reason": finish_reason,
                         "message": {"content": json.dumps(assessment)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30}}


class LlmCleaningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cleaner = LlmCleaner(CleaningConfig(), self.directory)

    def run_clean(self, blocks, responses):
        pending = iter(responses)

        def request(path, payload=None):
            if path == "/props":
                return {"default_generation_settings": {"n_ctx": 4096}, "build_info": "test"}
            self.assertEqual(path, "/v1/chat/completions")
            # An unspecified retry repeats the bad output, as a model may do.
            return next(pending, responses[-1])

        with patch.object(self.cleaner, "_prompt_tokens", return_value=100), \
                patch.object(self.cleaner, "_request", side_effect=request):
            return self.cleaner.clean(blocks, title="测试", provenance={"doc_id": "original"})

    def test_extracts_only_original_spans_and_records_exact_input(self):
        blocks = [{"id": "a", "text": "正常中文🙂。点击领取优惠！后续正文。"},
                  {"id": "b", "text": "整段广告"}]
        result = self.run_clean(blocks, [completion([1, 3])])
        self.assertEqual(result["edited_text"], "正常中文🙂。后续正文。")
        self.assertTrue(result["text_changed"])
        self.assertEqual(result["removals"][0]["text"], "点击领取优惠！")
        self.assertEqual(result["removals"][0]["start"], len("正常中文🙂。"))
        report = json.loads(next(self.directory.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["input_blocks"], blocks)
        self.assertEqual(report["provenance"], {"doc_id": "original"})
        self.assertEqual(report["result"], result)

    def test_uncertain_suggestions_do_not_delete_text(self):
        blocks = [{"id": "a", "text": "特殊主题正文"}]
        result = self.run_clean(blocks, [completion([0], decision="unsure")])
        self.assertEqual(result["decision"], "unsure")
        self.assertEqual(result["removals"], [])
        self.assertEqual(result["edited_text"], blocks[0]["text"] + "\n\n")
        self.assertFalse(result["text_changed"])

    def test_rejects_invalid_spans_truncated_output_and_partial_drop(self):
        for response in [completion([7]), completion([0, 0]), completion([0]),
                         completion([], decision="drop"), completion([], finish_reason="length"),
                         {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]}]:
            with self.subTest(response=response), self.assertRaises(LlmCleaningError):
                self.run_clean([{"id": "a", "text": "正文"}], [response])
            self.assertEqual(list(self.directory.iterdir()), [])

    def test_context_splitting_covers_entire_long_document_in_order(self):
        blocks = [{"id": "long", "text": "🙂长文章" * 1700},
                  {"id": "tail", "text": "末尾不能被截断。"}]
        units = split_units(blocks)
        # Force oversized single units to be split as well as oversized groups.
        def token_count(messages):
            return 600 + sum(len(item["text"]) for item in json.loads(messages[1]["content"])["units"])

        with patch.object(self.cleaner, "_prompt_tokens", side_effect=token_count):
            chunks = self.cleaner._plan(None, units, context=3072)
        self.assertGreater(len(chunks), 1)
        flattened = [unit for chunk in chunks for unit in chunk]
        for block in blocks:
            selected = [unit for unit in flattened if unit.block_id == block["id"]]
            self.assertEqual("".join(unit.text for unit in selected), block["text"])
            self.assertEqual(selected[0].start, 0)
            self.assertEqual(selected[-1].end, len(block["text"]))
            self.assertTrue(all(a.end == b.start for a, b in zip(selected, selected[1:])))
        for chunk in chunks:
            self.assertLessEqual(token_count(self.cleaner._messages(None, chunk)) + self.cleaner.config.max_output_tokens + 64, 3072)

    def test_sentence_split_round_trips_whitespace_emoji_and_urls(self):
        text = "标题\n\n中文🙂。广告！\nhttps://example.org/x?q=1.2\n尾部  "
        units = split_units([{"id": "a", "text": text}])
        self.assertEqual("".join(unit.text for unit in units), text)
        for unit in units:
            self.assertEqual(text[unit.start:unit.end], unit.text)

    def test_history_quotations_are_not_split_at_internal_questions_or_sentences(self):
        text = '含义\n 梁启超：“史者何？记述人类社会的活动。”\n《百科》：“第一，事件。第二，研究。”\n正文继续。'
        units = split_units([{"id": "history", "text": text}])
        self.assertEqual([unit.text for unit in units], [
            '含义\n', ' 梁启超：“史者何？记述人类社会的活动。”\n',
            '《百科》：“第一，事件。第二，研究。”\n', '正文继续。',
        ])
        self.assertEqual("".join(unit.text for unit in units), text)
        for unit in units:
            self.assertEqual(text[unit.start:unit.end], unit.text)
        unmatched = split_units([{"id": "broken", "text": '残缺“开头。下一句。'}])
        self.assertEqual([unit.text for unit in unmatched], ['残缺“开头。', '下一句。'])

    def test_invalid_cross_unit_edit_retries_only_that_chunk_with_feedback(self):
        blocks = [{"id": str(index), "text": "正文。"} for index in range(25)]
        bad = completion(edits=[{"unit_id": 0, "replacement": "抄入其他片段的文字"}])
        result = self.run_clean(blocks, [completion(), bad, completion()])
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["retry_count"], 1)
        self.assertFalse(result["text_changed"])
        self.assertEqual(result["input_tokens"], 300)
        report = json.loads(next(self.directory.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["retry_attempts"][0]["chunk"], 2)
        self.assertEqual(report["retry_attempts"][0]["response"], bad)
        self.assertIn("retry_instruction", json.loads(self.cleaner._messages(None, split_units(blocks[:1]), "无效修订")[1]["content"]))

    def test_failure_after_first_chunk_does_not_commit_partial_suggestion(self):
        blocks = [{"id": str(i), "text": "广告"} for i in range(25)]
        with self.assertRaises(LlmCleaningError):
            self.run_clean(blocks, [completion([0]), completion([99])])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_supplementary_material_is_removed_even_when_not_empty(self):
        blocks = [{"id": str(i), "text": text} for i, text in enumerate(
            ["条款", "条约正文。", "参见", "中英平等新约", "参考文献", "延伸阅读"])]
        response = completion([0, 2, 3, 4, 5])
        content = json.loads(response["choices"][0]["message"]["content"])
        for removal in content["removals"]:
            removal["reason"] = "link_list"
        response["choices"][0]["message"]["content"] = json.dumps(content)
        result = self.run_clean(blocks, [response])
        self.assertEqual([item["text"] for item in result["removals"]], ["条款", "参见", "中英平等新约", "参考文献", "延伸阅读"])
        self.assertEqual(result["edited_text"], "条约正文。")
        self.assertEqual(result["warnings"], [])

    def test_local_only_configuration_and_lock_release_after_failure(self):
        for url in ["https://example.com", "http://192.168.1.2:8080", "http://localhost:8080/v1"]:
            with self.assertRaises(ValueError):
                CleaningConfig(base_url=url)
        with self.assertRaises(LlmCleaningError):
            self.run_clean([{"id": "a", "text": "正文"}], [completion([4])])
        result = self.run_clean([{"id": "a", "text": "正文"}], [completion()])
        self.assertEqual(result["decision"], "keep")

    def test_crops_and_splices_fragments_without_losing_surrounding_paragraphs(self):
        blocks = [{"id": str(i), "text": text, "separator_after": "\n\n"}
                  for i, text in enumerate(["猫是哺乳动物，", "广告", "会喵喵叫（）。", "另一段正文。"])]
        result = self.run_clean(blocks, [completion([1], edits=[
            {"unit_id": 2, "replacement": "会喵喵叫。"},
        ], joins=[[0, 2]])])
        self.assertEqual(result["edited_text"], "猫是哺乳动物，会喵喵叫。\n\n另一段正文。")
        self.assertEqual(result["reordered_chunks"], 1)
        self.assertEqual(result["edits"][0]["original"], "会喵喵叫（）。")

    def test_partial_edit_and_paragraph_split(self):
        result = self.run_clean([{"id": "a", "text": "有用正文夹带广告。后半段正文。"}], [completion(
            edits=[{"unit_id": 0, "replacement": "有用正文。\n\n"}],
        )])
        self.assertEqual(result["edited_text"], "有用正文。\n\n后半段正文。")
        self.assertEqual(result["removals"], [])
        self.assertTrue(result["text_changed"])

    def test_reassembly_cannot_silently_lose_duplicate_or_invent_ids(self):
        for joins in [[[0]], [[0, 0, 1]], [[0, 1, 8]], [[], [0, 1]], [[False, 1]], [[1, 0]], [[0, 2]], [[0, 1], [1, 2]]]:
            with self.subTest(joins=joins), self.assertRaises(LlmCleaningError):
                self.run_clean([{"id": "a", "text": "第一句。第二句。第三句。"}], [completion(joins=joins)])
        with self.assertRaises(LlmCleaningError):
            self.run_clean([{"id": "a", "text": "正文。"}], [completion([0], edits=[{"unit_id": 0, "replacement": "冲突"}])])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_noop_and_unsure_preserve_exact_draft_separators(self):
        blocks = [{"id": "a", "text": "正文🙂。", "separator_after": "\r\n\t\r\n"},
                  {"id": "b", "text": "结尾  ", "separator_after": ""}]
        original = "".join(block["text"] + block["separator_after"] for block in blocks)
        for response in [completion(), completion(decision="unsure", edits=[{"unit_id": 0, "replacement": "错误"}], joins=[[1, 0]])]:
            result = self.run_clean(blocks, [response])
            self.assertEqual(result["edited_text"], original)
            self.assertFalse(result["text_changed"])
            self.assertEqual(result["edits"], [])

    def test_rejects_invented_words_and_intra_unit_reordering(self):
        for replacement in ["猫是一种优秀的动物。", "动物是猫。", "猫是猫是动物。"]:
            with self.subTest(replacement=replacement), self.assertRaises(LlmCleaningError):
                self.run_clean([{"id": "a", "text": "猫是动物。"}], [completion(
                    edits=[{"unit_id": 0, "replacement": replacement}],
                )])

    def test_section_context_survives_chunk_boundaries_and_table_exit(self):
        blocks = [{"id": "a", "text": '{| style="x"\n| 数学逻辑 || 集合论\n|}\n正文。\n参考文献\n' + 'Book entry\n' * 30}]
        units = split_units(blocks)
        self.assertEqual(units[1].section_hint, "Wiki表格")
        self.assertIsNone(units[3].section_hint)
        with patch.object(self.cleaner, "_prompt_tokens", return_value=100):
            chunks = self.cleaner._plan(None, units, 4096)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[-1][0].section_hint, "参考文献")


if __name__ == "__main__":
    unittest.main()
