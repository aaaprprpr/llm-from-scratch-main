"""Download raw sources separately; encode completed Dataset/Arrow directories."""

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from datasets import Dataset, Features, Value

from dataset.data_pipeline import build_bin, download
from dataset.data_pipeline.config import validate_config
from dataset.label.backend.dataset_store import write_arrow_dataset
from tokenizer import Tokenizer
from train.pretrain.train_model import load_token_bin

TOKENIZER_PATH = ROOT / "tokenizer" / "bpe_8192"


def bin_settings(input_dataset, train_bin, val_bin, workers=1):
    return {
        "build_bin": {
            "input": str(input_dataset),
            "tokenizer": str(TOKENIZER_PATH),
            "train_bin": str(train_bin),
            "val_bin": str(val_bin),
            "train_ratio": 0.75,
            "seed": 42,
            "workers": workers,
            "overwrite": False,
        },
    }


class SourcePipelineTests(unittest.TestCase):
    def test_download_source_config(self):
        source = {
            "kind": "jsonl",
            "repo": "jingyaogong/minimind_dataset",
            "revision": "test",
            "filename": "sft_t2t_mini.jsonl",
            "path": "raw/sft_t2t_mini.jsonl",
            "sha256": "0" * 64,
        }
        config = {
            "sources": {"minimind": source},
            "download": {"sources": [], "cleanup_cache": False},
        }
        with self.assertRaisesRegex(ValueError, "SFT/RL"):
            validate_config(config)
        source["filename"] = "pretrain_t2t.jsonl"
        source["path"] = "raw/pretrain_t2t.jsonl"
        validate_config(config)
        source["unused_note"] = "ignored previously"
        with self.assertRaisesRegex(ValueError, "unused"):
            validate_config(config)

    def test_dataset_directory_preserves_records_and_worker_order(self):
        texts = [
            "  第一段\n\n第二段\t  ",
            "<p>原文</p> https://example.com/?a=1&b=2",
            "这段保留重复。",
            "这段保留重复。",
            "甲乙丙丁。" * 1600,
            "English and 中文🙂 café e\u0301Ａ",
            "最后一条记录\r\n第二行",
            "\t \n  ",
        ]
        tokenizer = Tokenizer(str(TOKENIZER_PATH))
        eos = tokenizer.special_token_to_id[build_bin.EOS_TOKEN]
        expected = Counter(tuple(tokenizer.encode(text)) for text in texts)
        self.assertGreater(max(map(len, expected)), 2048)
        tokenizer_sha = build_bin.tokenizer_fingerprint(TOKENIZER_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dataset = root / "completed_dataset"
            Dataset.from_dict({"text": texts}).save_to_disk(str(input_dataset))
            fingerprint = build_bin.load_input_dataset(input_dataset)._fingerprint
            outputs, metadata_pairs = [], []
            for workers in (1, 2):
                train = root / f"train{workers}.bin"
                val = root / f"val{workers}.bin"
                config = bin_settings(input_dataset, train, val, workers)
                config_path = root / "build_bin.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with (
                    patch.object(build_bin, "CONFIG_PATH", config_path),
                    patch.object(build_bin, "BATCH_RECORDS", 2),
                    patch.object(build_bin, "SHUFFLE_BLOCK_RECORDS", 3),
                ):
                    metadata = build_bin.main()
                    with self.assertRaises(FileExistsError):
                        build_bin.main()
                self.assertEqual([m["records"] for m in metadata], [6, 2])
                observed = Counter()
                pair = []
                comparable_metadata = []
                for path, meta in zip((train, val), metadata):
                    self.assertEqual(meta["input_dataset"], str(input_dataset.resolve()))
                    self.assertEqual(meta["input_records"], len(texts))
                    self.assertEqual(meta["dataset_fingerprint"], fingerprint)
                    self.assertEqual(meta["tokenizer_sha256"], tokenizer_sha)
                    self.assertEqual(meta["eos_id"], eos)
                    self.assertEqual(meta["workers"], workers)
                    tokens = load_token_bin(
                        path,
                        expected_tokenizer_size=len(tokenizer.tokenizer),
                        expected_tokenizer_sha256=tokenizer_sha,
                    )
                    try:
                        ids = tokens.tolist()
                        self.assertEqual(tokens.dtype, np.dtype("uint16"))
                        self.assertEqual(len(ids), meta["tokens"])
                        stops = [i for i, token_id in enumerate(ids) if token_id == eos]
                        self.assertEqual(len(stops), meta["records"])
                        start = 0
                        for stop in stops:
                            observed[tuple(ids[start:stop])] += 1
                            start = stop + 1
                    finally:
                        tokens._mmap.close()
                    pair.append(path.read_bytes())
                    comparable_metadata.append({
                        key: value for key, value in meta.items()
                        if key not in {"workers", "output_bin"}
                    })
                self.assertEqual(observed, expected)
                outputs.append(pair)
                metadata_pairs.append(comparable_metadata)
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(metadata_pairs[0], metadata_pairs[1])

    def test_arrow_directory_is_bin_input(self):
        texts = ["  保留正文和空白\n🙂  ", "这段重复。" * 10]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dataset = root / "completed_arrow"
            write_arrow_dataset(
                ({"text": text, "doc_id": str(index)} for index, text in enumerate(texts)),
                input_dataset,
                features=Features({"text": Value("string"), "doc_id": Value("string")}),
                fingerprint="completed-arrow-fixture",
            )
            loaded = build_bin.load_input_dataset(input_dataset)
            self.assertEqual(loaded[:]["text"], texts)
            self.assertEqual(loaded._fingerprint, "completed-arrow-fixture")
            del loaded
            train, val = root / "train.bin", root / "val.bin"
            config_path = root / "build_bin.json"
            config_path.write_text(
                json.dumps(bin_settings(input_dataset, train, val)), encoding="utf-8"
            )
            with patch.object(build_bin, "CONFIG_PATH", config_path):
                metadata = build_bin.main()
            self.assertEqual([item["records"] for item in metadata], [1, 1])
            self.assertTrue(train.is_file() and val.is_file())

    def test_restricted_hf_subsets_remain_guarded(self):
        source = {
            "repo": "opencsg/Fineweb-Edu-Chinese-V2.1",
            "split": "train",
            "data_files": {"train": "*/*.parquet"},
        }
        with self.assertRaisesRegex(ValueError, "Unsafe file selection"):
            download._guard_dataset(source)
        source["data_files"] = {"train": "4_5/*.parquet"}
        download._guard_dataset(source)
        source = {"repo": "wikimedia/wikipedia", "config": None, "split": "train"}
        with self.assertRaisesRegex(ValueError, "Unsafe subset"):
            download._guard_dataset(source)

    def test_existing_downloads_and_mock_jsonl_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "pretrain_t2t.jsonl"
            raw.write_text(
                '{"text":"第一条记录。"}\n{"text":"第二条记录。"}\n', encoding="utf-8"
            )
            source = {
                "source_id": "minimind",
                "kind": "jsonl",
                "path": str(raw),
                "repo": "jingyaogong/minimind_dataset",
                "revision": "test",
                "filename": raw.name,
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            }
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

            with patch.object(download, "SAMPLE_DIR", root / "samples"), patch(
                "huggingface_hub.hf_hub_download", side_effect=fake_hub_download
            ):
                self.assertEqual(
                    download.download_one({**source, "path": str(remote)})["status"],
                    "downloaded",
                )
            self.assertEqual(remote.read_bytes(), raw.read_bytes())

            raw_dataset = root / "hf"
            Dataset.from_dict({"text": ["第一条记录。", "第二条记录。"]}).save_to_disk(
                str(raw_dataset)
            )
            with patch.object(download, "SAMPLE_DIR", root / "samples"):
                self.assertEqual(download.download_one({
                    "source_id": "sample", "kind": "hf_dataset", "repo": "example/local",
                    "split": "train", "path": str(raw_dataset),
                })["status"], "already_downloaded")


if __name__ == "__main__":
    unittest.main()
