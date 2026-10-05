from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from dataset.label.backend.api import create_app
from dataset.label.backend.batch_clean import clean_queue
from dataset.label.tests.test_batch_clean import _make_batch_queue

from dataset.label.backend.laya_cleaning import LayaBatcher, LayaCleaner
from dataset.label.backend.llm_cleaning import create_cleaner
from dataset.label.backend.model_settings import ModelSettings


class _FakeAgent:
    def predict_batch(self, states, questions, **kwargs):
        assert list(questions) == ["body"]
        return [{"answers": {"body": {"choice": "drop" if state == "参考文献" else "keep"}}}
                for state in states]


class LayaCleaningTests(unittest.TestCase):
    def test_cuda_model_disables_unbounded_shape_graph_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            (model_path / "model.safetensors").touch()
            agent = SimpleNamespace(device=SimpleNamespace(type="cuda"), cfg={"max_len": 1024},
                                    accelerate=Mock())
            fake_laya = ModuleType("laya")
            fake_laya.load = Mock(return_value=agent)
            with (patch.object(LayaCleaner, "_agent", None),
                  patch("dataset.label.backend.laya_cleaning.MODEL_DIRECTORY", model_path),
                  patch.dict(sys.modules, {"laya": fake_laya}),
                  patch("torch.cuda.is_available", return_value=True)):
                self.assertIs(LayaCleaner._model(), agent)
            self.assertEqual(agent.cfg["max_len"], 2048)
            agent.accelerate.assert_called_once_with(use_graphs=False, strict=False)

    def test_cpu_fallback_restores_fast_path_without_cuda_graphs(self):
        import torch
        from laya.agent import Agent

        agent = Agent.__new__(Agent)
        agent.model = SimpleNamespace(to=Mock())
        agent.accelerate = Mock()
        agent._restore_runtime(torch.device("cuda"), torch.bfloat16, True, False)
        agent.model.to.assert_called_once_with(torch.device("cuda"))
        agent.accelerate.assert_called_once_with(use_graphs=False)

    def test_fast_inference_rechecks_only_uncertain_answers(self):
        class FastAgent:
            _fast = object()

            def __init__(self):
                self.model = type("Model", (), {})()
                self._stock_forward = self.stock_forward
                self.model.forward = self.fast_forward
                self.calls = []

            def fast_forward(self):
                pass

            def stock_forward(self):
                pass

            def predict_batch(self, states, questions, **kwargs):
                stock = self.model.forward == self._stock_forward
                self.calls.append((list(states), stock))
                return [{"answers": {"body": {
                    "choice": "keep" if stock or state == "clear" else "drop",
                    "answer_confidence": 0.51 if state == "uncertain" else 0.98,
                }}} for state in states]

        agent = FastAgent()
        with patch.object(LayaCleaner, "_model", return_value=agent):
            answers = LayaCleaner._predict_states(["clear", "uncertain"])
        self.assertEqual([answer["answers"]["body"]["choice"] for answer in answers],
                         ["keep", "keep"])
        self.assertEqual(agent.calls, [(["clear", "uncertain"], False), (["clear", "uncertain"], True)])
        self.assertEqual(agent.model.forward, agent.fast_forward)

    def test_failed_fast_kernel_falls_back_to_stock_model(self):
        class FailingFastAgent:
            def __init__(self):
                self._fast = object()
                self.calls = 0

            def predict_batch(self, states, questions, **kwargs):
                self.calls += 1
                if self._fast is not None:
                    raise RuntimeError("cuda_runtime.h missing")
                return [{"answers": {"body": {"choice": "keep"}}} for _ in states]

            def deaccelerate(self):
                self._fast = None

        agent = FailingFastAgent()
        with patch.object(LayaCleaner, "_model", return_value=agent):
            answers = LayaCleaner._predict_states(["正文。"])
        self.assertEqual(answers[0]["answers"]["body"]["choice"], "keep")
        self.assertEqual(agent.calls, 2)
        self.assertIsNone(agent._fast)

    def test_model_selection_and_existing_review_assembly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ModelSettings(root).config("laya")
            self.assertEqual(config.provider, "laya")
            cleaner = create_cleaner(config, root / "reports")
            self.assertIsInstance(cleaner, LayaCleaner)
            blocks = [
                {"id": "a", "text": "参考文献", "separator_after": "\n"},
                {"id": "b", "text": "数学研究数量和结构。", "separator_after": "\n"},
            ]
            with patch.object(LayaCleaner, "_model", return_value=_FakeAgent()):
                result = cleaner.clean(blocks, title="数学", provenance={"doc_id": "doc", "queue_id": "queue"})
            self.assertEqual(result["decision"], "keep")
            self.assertEqual(result["edited_text"], "数学研究数量和结构。")
            self.assertEqual([item["text"] for item in result["removals"]], ["参考文献"])
            self.assertTrue(cleaner.is_complete(result))
            cached = cleaner.clean(blocks, title="数学", provenance={"doc_id": "doc", "queue_id": "queue"})
            self.assertIn("已载入上次完成", cached["warnings"][0])


    def test_newline_block_is_one_decision_unit(self):
        cleaner = create_cleaner(ModelSettings(Path("/tmp")).config("laya"), Path("/tmp/laya-probe"))
        blocks = [
            {"id": "one", "text": "第一句。第二句。"},
            {"id": "two", "text": "参考文献"},
        ]
        units = cleaner._split_units(blocks)
        self.assertEqual([(unit.block_id, unit.start, unit.end, unit.text) for unit in units], [
            ("one", 0, len(blocks[0]["text"]), blocks[0]["text"]),
            ("two", 0, len(blocks[1]["text"]), blocks[1]["text"]),
        ])
        self.assertGreater(len(cleaner._split_units([{"id": "long", "text": "正文。" * 401}])), 1)

    def test_generative_sources_keep_their_sentence_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = ModelSettings(root)
            block = [{"id": "one", "text": "第一句。第二句。"}]
            deepseek = create_cleaner(settings.config("deepseek"), root / "reports")
            laya = create_cleaner(settings.config("laya"), root / "reports")
            self.assertEqual([unit.text for unit in deepseek._split_units(block)],
                             ["第一句。", "第二句。"])
            self.assertEqual([unit.text for unit in laya._split_units(block)],
                             ["第一句。第二句。"])

    def test_batcher_merges_concurrent_requests_without_mixing_answers(self):
        class CountingAgent(_FakeAgent):
            def __init__(self):
                self.calls = []

            def predict_batch(self, states, questions, **kwargs):
                self.calls.append(list(states))
                return super().predict_batch(states, questions, **kwargs)

        agent = CountingAgent()
        barrier = threading.Barrier(4)
        with patch.object(LayaCleaner, "_model", return_value=agent):
            with LayaBatcher(gather_seconds=0.05) as batcher:
                def request(value):
                    barrier.wait()
                    return batcher.predict([value])

                with ThreadPoolExecutor(max_workers=4) as pool:
                    answers = list(pool.map(request, ["正文一。", "参考文献", "正文二。", "正文三。"]))
        self.assertEqual([answer[0]["answers"]["body"]["choice"] for answer in answers],
                         ["keep", "drop", "keep", "keep"])
        self.assertLess(len(agent.calls), 4)
        self.assertEqual(sum(map(len, agent.calls)), 4)

    def test_batch_queue_pause_resume_keeps_results_and_reports(self):
        class StoppingAgent(_FakeAgent):
            def __init__(self, stop):
                self.stop = stop
                self.calls = []

            def predict_batch(self, states, questions, **kwargs):
                self.calls.append(list(states))
                self.stop.set()
                return super().predict_batch(states, questions, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = [f"正文第 {index} 条。\n参考文献" for index in range(8)]
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, texts)
            cleaner = create_cleaner(ModelSettings(root).config("laya"), root / "reports")
            stop = threading.Event()
            agent = StoppingAgent(stop)
            options = dict(database_path=root / "curation.sqlite3", queue_id=queue,
                           output_directory=root / "batch", cleaner=cleaner,
                           cleaners={"laya": cleaner}, failure_fallback=None)
            with patch.object(LayaCleaner, "_model", return_value=agent):
                paused = clean_queue(**options, stop_event=stop)
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(paused["source_activity"]["laya"]["capacity"], 4)
                self.assertEqual(paused["workers"], 4)
                self.assertLess(paused["processed"], 8)
                resumed = clean_queue(**options)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["counts"], {"keep": 8})
            rows = [json.loads(line) for line in (root / "batch" / "progress.jsonl").read_text().splitlines()]
            self.assertEqual({row["ordinal"] for row in rows}, set(range(8)))
            output = [json.loads(line)["text"] for line in
                      (root / "batch" / "cleaned.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(output), sorted(f"正文第 {index} 条。" for index in range(8)))
            self.assertEqual(len(list((root / "reports").rglob("*.json"))), 8)

    def test_batch_worker_keeps_laya_cleaner_type(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with TestClient(create_app(root)) as client:
                queue = _make_batch_queue(client, root, ["参考文献\n数学研究数量和结构。"])
            cleaner = create_cleaner(ModelSettings(root).config("laya"), root / "reports")
            with patch.object(LayaCleaner, "_model", return_value=_FakeAgent()):
                manifest = clean_queue(
                    database_path=root / "curation.sqlite3", queue_id=queue,
                    output_directory=root / "batch", cleaner=cleaner,
                    cleaners={"laya": cleaner}, failure_fallback=None, workers=1, limit=1,
                )
            self.assertEqual(manifest["source_activity"]["laya"]["capacity"], 4)
            self.assertEqual(manifest["counts"], {"keep": 1})
            self.assertIn("数学研究数量和结构。", (root / "batch" / "cleaned.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
