"""The same source catalog drives download, optional cleanup, and bin encoding."""

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from datasets import Dataset

from dataset.data_pipeline import build_bin, download, preprocess
from dataset.data_pipeline.config import validate_config
from dataset.data_pipeline.jsonl import JsonlTextDataset
from tokenizer import Tokenizer
from train.pretrain.train_model import load_token_bin

ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_PATH = ROOT / "tokenizer" / "bpe_8192"


def settings(sources, source_name, train_bin, val_bin, *, preprocess_sources=None, prepared_path=None):
    return {
        "sources": sources,
        "download": {"sources": [], "cleanup_cache": False},
        "preprocess": {"sources": preprocess_sources or [],
                       "output": str(prepared_path or train_bin.parent / "prepared"),
                       "fix_text": True, "max_repetition_ratio": 0.8,
                       "workers": 1, "batch_size": 2, "overwrite": False},
        "build_bin": {"input": source_name, "tokenizer": str(TOKENIZER_PATH),
                      "train_bin": str(train_bin), "val_bin": str(val_bin),
                      "train_ratio": 0.75, "seed": 42, "workers": 1,
                      "overwrite": False},
    }


class SourcePipelineTests(unittest.TestCase):
    def test_jsonl_schema_and_source_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            for bad in (b"not json", b'{"conversations": []}', b'{"text": 42}', b'{"text": ""}', b"\n"):
                with self.subTest(bad=bad):
                    path.write_bytes(b'{"text": "valid"}\n' + bad + b"\n")
                    with self.assertRaisesRegex(ValueError, r"bad.jsonl:2"):
                        JsonlTextDataset(path)
            source = {"kind": "jsonl", "repo": "jingyaogong/minimind_dataset",
                      "revision": "test", "filename": "sft_t2t_mini.jsonl",
                      "path": str(path), "sha256": "0" * 64}
            config = settings({"minimind": source}, "minimind", path.with_name("train.bin"), path.with_name("val.bin"))
            with self.assertRaisesRegex(ValueError, "SFT/RL"):
                validate_config(config)
            source["filename"] = "pretrain_t2t.jsonl"
            source["path"] = str(path.with_name(source["filename"]))
            source["unused_note"] = "ignored previously"
            with self.assertRaisesRegex(ValueError, "unused"):
                validate_config(config)

    def test_jsonl_uses_common_builder_and_preserves_existing_bins(self):
        texts = [
            "  第一段\n\n第二段\t  ", "<p>原文</p> https://example.com/?a=1&b=2",
            "这段保留重复。", "这段保留重复。", "甲乙丙丁。" * 1600,
            "English and 中文🙂", "最后一条记录", "\t \n  ",
        ]
        tokenizer = Tokenizer(str(TOKENIZER_PATH))
        eos = tokenizer.special_token_to_id[build_bin.EOS_TOKEN]
        expected = Counter(tuple(tokenizer.encode(text)) for text in texts)
        self.assertGreater(max(map(len, expected)), 2048)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "pretrain_t2t.jsonl"
            raw.write_text("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts) + "\n", encoding="utf-8")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            source = {"kind": "jsonl", "repo": "jingyaogong/minimind_dataset",
                      "revision": "test", "filename": raw.name, "sha256": digest, "path": str(raw)}
            outputs = []
            for workers in (1, 2):
                train = root / f"train{workers}.bin"
                val = root / f"val{workers}.bin"
                config = settings({"minimind": source}, "minimind", train, val)
                config["build_bin"]["workers"] = workers
                config_path = root / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with patch("dataset.data_pipeline.config.CONFIG_PATH", config_path):
                    metadata = build_bin.main()
                    with self.assertRaises(FileExistsError):
                        build_bin.main()
                self.assertEqual([m["records"] for m in metadata], [6, 2])
                observed = Counter()
                pair = []
                for path, meta in zip((train, val), metadata):
                    tokens = load_token_bin(path, expected_tokenizer_size=len(tokenizer.tokenizer),
                                            expected_tokenizer_sha256=meta["tokenizer_sha256"])
                    try:
                        ids = tokens.tolist()
                        self.assertEqual(tokens.dtype, np.dtype("uint16"))
                        self.assertEqual(len(ids), meta["tokens"])
                        self.assertEqual(meta["dataset_fingerprint"], digest)
                        stops = [i for i, token_id in enumerate(ids) if token_id == eos]
                        self.assertEqual(len(stops), meta["records"])
                        start = 0
                        for stop in stops:
                            observed[tuple(ids[start:stop])] += 1
                            start = stop + 1
                    finally:
                        tokens._mmap.close()
                    pair.append(path.read_bytes())
                self.assertEqual(observed, expected)
                outputs.append(pair)
            self.assertEqual(outputs[0], outputs[1])
            source["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                JsonlTextDataset(raw, expected_sha256=source["sha256"])

    def test_restricted_hf_subsets_remain_guarded(self):
        source = {"repo": "opencsg/Fineweb-Edu-Chinese-V2.1", "split": "train",
                  "data_files": {"train": "*/*.parquet"}}
        with self.assertRaisesRegex(ValueError, "Unsafe file selection"):
            download._guard_dataset(source)
        source["data_files"] = {"train": "4_5/*.parquet"}
        download._guard_dataset(source)
        source = {"repo": "wikimedia/wikipedia", "config": None, "split": "train"}
        with self.assertRaisesRegex(ValueError, "Unsafe subset"):
            download._guard_dataset(source)

    def test_existing_jsonl_download_and_preprocessed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "pretrain_t2t.jsonl"
            raw.write_text('{"text":"第一条记录。"}\n{"text":"第二条记录。"}\n', encoding="utf-8")
            source = {"source_id": "minimind", "kind": "jsonl", "path": str(raw),
                      "repo": "jingyaogong/minimind_dataset", "revision": "test",
                      "filename": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
            with patch.object(download, "SAMPLE_DIR", root / "samples"):
                self.assertEqual(download.download_one(source)["status"], "already_downloaded")
            self.assertTrue((root / "samples" / "minimind.sample.json").is_file())

            remote = root / "remote" / raw.name
            def fake_hub_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], source["repo"])
                self.assertEqual(kwargs["filename"], source["filename"])
                self.assertEqual(kwargs["revision"], source["revision"])
                remote.parent.mkdir(exist_ok=True)
                remote.write_bytes(raw.read_bytes())
                return str(remote)
            with patch.object(download, "SAMPLE_DIR", root / "samples"), \
                 patch("huggingface_hub.hf_hub_download", side_effect=fake_hub_download):
                self.assertEqual(download.download_one({**source, "path": str(remote)})["status"], "downloaded")
            self.assertEqual(remote.read_bytes(), raw.read_bytes())

            raw_dataset = root / "hf"
            Dataset.from_dict({"text": [
                "这是一段关于自然语言处理的完整中文资料。",
                "另一个样本描述了机器学习模型的训练过程。",
                "作者在文章中介绍了图书馆和读者之间的关系。",
                "程序员记录了天气变化和城市交通的观察。",
            ]}).save_to_disk(str(raw_dataset))
            train, val = root / "prepared_train.bin", root / "prepared_val.bin"
            prepared = root / "prepared"
            config = settings({"sample": {"kind": "hf_dataset", "repo": "example/local",
                                           "split": "train", "path": str(raw_dataset), "adapter": "text_only"}},
                              "preprocess", train, val, preprocess_sources=["sample"], prepared_path=prepared)
            with patch.object(download, "SAMPLE_DIR", root / "samples"):
                self.assertEqual(download.download_one({"source_id": "sample", **config["sources"]["sample"]})["status"], "already_downloaded")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch("dataset.data_pipeline.config.CONFIG_PATH", config_path):
                preprocess.main()
                train_meta, val_meta = build_bin.main()
            self.assertTrue(prepared.is_dir())
            self.assertEqual(train_meta["records"] + val_meta["records"], 4)
            self.assertTrue(train.is_file() and val.is_file())


if __name__ == "__main__":
    unittest.main()
