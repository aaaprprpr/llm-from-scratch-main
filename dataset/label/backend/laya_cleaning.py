"""Optional local Laya decision model using the existing deletion-only review pipeline."""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future
from queue import Empty, Queue

from ..laya import CLEANING_MODEL_DIRECTORY
from .llm_cleaning import (ChunkAssessment, LlmCleaner, LlmCleaningError, Removal,
                           SourceUnavailableError, TextUnit, split_units)

MODEL_DIRECTORY = CLEANING_MODEL_DIRECTORY
# TileLang can flip choices very close to 0.5; recheck those with the stock forward.
FAST_RECHECK_CONFIDENCE = 0.55
logger = logging.getLogger(__name__)
QUESTION = {
    "body": {
        "type": "choice",
        "instructions": "这段文字是否适合作为预训练语料中的正文保留？",
        "criteria": {
            "keep": "能阅读的事实叙述、解释、定义、过程、观点、诗歌或完整代码。",
            "drop": "只有标题、日期、导航、名字清单、字段值、表格残片、书目信息、占位模板。",
        },
    }
}


class LayaBatcher:
    """Merge pending Laya chunks while batch workers handle separate documents."""

    def __init__(self, target_units: int = 128, gather_seconds: float = 0.002):
        self.target_units = target_units
        self.gather_seconds = gather_seconds
        self._requests: Queue[tuple[list[str], Future] | None] = Queue()
        self._closed = False
        self._gate = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="laya-batch-inference")
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def predict(self, states: list[str]) -> list[dict]:
        future: Future = Future()
        with self._gate:
            if self._closed:
                raise LlmCleaningError("Laya 批量推理已经结束")
            self._requests.put((states, future))
        return future.result()

    def close(self):
        with self._gate:
            if self._closed:
                return
            self._closed = True
            self._requests.put(None)
        self._thread.join()

    def _run(self):
        while True:
            first = self._requests.get()
            if first is None:
                return
            group = [first]
            count = len(first[0])
            deadline = time.monotonic() + self.gather_seconds
            stopping = False
            while count < self.target_units:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    following = self._requests.get(timeout=remaining)
                except Empty:
                    break
                if following is None:
                    stopping = True
                    break
                group.append(following)
                count += len(following[0])
            states = [state for part, _ in group for state in part]
            try:
                answers = LayaCleaner._predict_states(states)
                if len(answers) != len(states):
                    raise LlmCleaningError("Laya 返回的片段数量与原文不一致")
            except Exception as exc:
                for _, future in group:
                    future.set_exception(exc)
            else:
                offset = 0
                for part, future in group:
                    future.set_result(answers[offset:offset + len(part)])
                    offset += len(part)
            if stopping:
                return


class LayaCleaner(LlmCleaner):
    """Classify each newline block, then let LlmCleaner assemble and cache the result."""

    _agent = None
    _agent_lock = threading.Lock()
    _inference_lock = threading.Lock()

    @staticmethod
    def _split_units(blocks: list[dict]) -> list[TextUnit]:
        # A line block is the classification target. Split only oversized blocks so
        # they fit within the model input and the existing span-based assembly.
        units = []
        for block in blocks:
            text = block["text"]
            units.extend(split_units([block]) if len(text) > 1200 else
                         [TextUnit(block["id"], 0, len(text), text)])
        return units

    def _messages(self, title: str | None, units: list[TextUnit], retry_error: str | None = None) -> list[dict]:
        # LlmCleaner stores this string as report metadata/cache identity. Laya
        # receives only QUESTION and each unit's text, never chat messages.
        metadata = {"question": QUESTION, "segmentation": "newline-block-v1; split if over 1200 chars"}
        return [{"role": "system", "content": json.dumps(metadata, ensure_ascii=False, sort_keys=True)}]

    def _prompt_tokens(self, messages: list[dict]) -> int:
        # Laya reads each unit separately; the shared question does not consume a document window.
        return 0

    @classmethod
    def _model(cls):
        with cls._agent_lock:
            if cls._agent is None:
                if not (MODEL_DIRECTORY / "model.safetensors").is_file():
                    raise LlmCleaningError(f"缺少本地 Laya 权重：{MODEL_DIRECTORY}")
                try:
                    import torch
                    from ..laya.runtime import load
                    agent = load(str(MODEL_DIRECTORY), device="cuda" if torch.cuda.is_available() else "cpu")
                    if agent.device.type == "cuda":
                        agent.cfg["max_len"] = max(int(agent.cfg.get("max_len", 0)), 2048)
                        # Lengths vary continuously; the unlimited CUDA graph cache
                        # eventually exhausts VRAM in a long cleaning run.
                        agent.accelerate(use_graphs=False, strict=False)
                    cls._agent = agent
                except Exception as exc:
                    raise SourceUnavailableError(f"Laya 模型加载失败：{exc}") from exc
            return cls._agent

    @classmethod
    def _predict_states(cls, states: list[str]) -> list[dict]:
        # Both batch and interactive calls use this lock, including the stock recheck.
        with cls._inference_lock:
            agent = cls._model()
            options = {"batch_size": 16, "sort_by_length": len(states) > 16, "max_len": 2048}
            try:
                answers = agent.predict_batch(states, QUESTION, **options)
            except Exception as exc:
                if getattr(agent, "_fast", None) is None:
                    raise
                agent.deaccelerate()
                logger.warning("Laya TileLang 推理失败，已切回 PyTorch：%s", str(exc)[-300:])
                answers = agent.predict_batch(states, QUESTION, **options)
            if getattr(agent, "_fast", None) is not None:
                uncertain = [index for index, answer in enumerate(answers)
                             if answer.get("answers", {}).get("body", {}).get("answer_confidence", 0)
                             < FAST_RECHECK_CONFIDENCE]
                if uncertain:
                    fast_forward = agent.model.forward
                    try:
                        agent.model.forward = agent._stock_forward
                        # Keep the original batch shape: rechecking a subset changes
                        # padding and can flip decisions right on the boundary.
                        verified = agent.predict_batch(states, QUESTION, **options)
                    finally:
                        agent.model.forward = fast_forward
                    for index in uncertain:
                        answers[index] = verified[index]
            return answers

    def _assess_chunk(self, title, units, context, chunk_number, chunk_count):
        indexed = [(index, unit.text.strip()) for index, unit in enumerate(units) if unit.text.strip()]
        removals = []
        if indexed:
            try:
                states = [value for _, value in indexed]
                batcher = getattr(self, "batch_dispatcher", None)
                answers = self._predict_states(states) if batcher is None else batcher.predict(states)
            except LlmCleaningError:
                raise
            except RuntimeError as exc:
                raise SourceUnavailableError(
                    f"Laya 第 {chunk_number}/{chunk_count} 块推理运行故障：{exc}"
                ) from exc
            except Exception as exc:
                raise LlmCleaningError(f"Laya 第 {chunk_number}/{chunk_count} 块推理失败：{exc}") from exc
            if len(answers) != len(indexed):
                raise LlmCleaningError("Laya 返回的片段数量与原文不一致")
            for (index, _), answer in zip(indexed, answers):
                choice = answer.get("answers", {}).get("body", {}).get("choice")
                if choice == "drop":
                    removals.append(Removal(unit_id=index, reason="non_prose"))
                elif choice != "keep":
                    raise LlmCleaningError("Laya 返回了无法识别的分类结果")
        assessment = ChunkAssessment(
            decision="drop" if len(removals) == len(units) else "keep",
            removals=removals, edits=[], joins=[],
            summary=f"Laya 判定删除 {len(removals)}/{len(units)} 个片段。",
        )
        return assessment, 0, 0, [], None
