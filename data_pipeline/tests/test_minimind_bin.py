import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from config_loader import Config
from data_pipeline.build_minimind_bin import CONFIG_PATH, JsonlTextDataset, obtain_jsonl, run
from pretrain.train_model import load_token_bin
from tokenizer import Tokenizer


class MiniMindBinTests(unittest.TestCase):
    def test_schema_errors_fail_instead_of_silently_dropping_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            for bad in (b"not json", b'{"conversations": []}', b'{"text": 42}', b'{"text": ""}', b"\n"):
                with self.subTest(bad=bad):
                    path.write_bytes(b'{"text": "valid"}\n' + bad + b"\n")
                    with self.assertRaisesRegex(ValueError, r"bad.jsonl:2"):
                        JsonlTextDataset(path)

    def test_source_hash_and_pretraining_file_selection(self):
        config = dict(Config(CONFIG_PATH).require("minimind_bin"))
        config["filename"] = "sft_t2t_mini.jsonl"
        with self.assertRaisesRegex(ValueError, "SFT/RL"):
            obtain_jsonl(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain_t2t_mini.jsonl"
            path.write_text('{"text":"one"}\n{"text":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                JsonlTextDataset(path, expected_sha256="0" * 64)

    def test_full_records_survive_and_bins_load_with_training_tokenizer(self):
        # Includes whitespace, HTML, URLs, duplicates and a record over 2048 tokens:
        # these must all be encoded as supplied, with no cleaning or truncation.
        texts = [
            "  第一段\n\n第二段\t  ", "<p>原文</p> https://example.com/?a=1&b=2",
            "这段保留重复。", "这段保留重复。", "甲乙丙丁。" * 1600,
            "English and 中文🙂", "最后一条记录", "\t \n  ",
        ]
        config = dict(Config(CONFIG_PATH).require("minimind_bin"))
        tokenizer = Tokenizer(str(Path(__file__).resolve().parents[2] / config["tokenizer"]))
        eos = tokenizer.special_token_to_id[config["eos_token"]]
        expected = Counter(tuple(tokenizer.encode(text)) for text in texts)
        self.assertGreater(max(map(len, expected)), 2048)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pretrain_t2t_mini.jsonl"
            path.write_text("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts) + "\n", encoding="utf-8")
            dataset = JsonlTextDataset(path)
            self.assertEqual(dataset[:]["text"], texts)
            self.assertEqual(dataset[2:5]["text"], texts[2:5])
            config.update(
                raw_dir=str(root), download=False, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                train_ratio=0.75, shuffle_block_records=3, batch_records=2, overwrite=False,
            )
            outputs = []
            for workers in (1, 2):
                config.update(workers=workers, train_bin=str(root / f"train{workers}.bin"), val_bin=str(root / f"val{workers}.bin"))
                metadata = run(config)
                self.assertEqual([m["records"] for m in metadata], [6, 2])
                observed = Counter()
                pair = []
                for key, meta in zip(("train_bin", "val_bin"), metadata):
                    tokens = load_token_bin(config[key], expected_tokenizer_size=24576, expected_tokenizer_sha256=meta["tokenizer_sha256"])
                    try:
                        ids = tokens.tolist()
                        self.assertEqual(tokens.dtype, np.dtype("uint16"))
                        self.assertEqual(len(ids), meta["tokens"])
                        self.assertEqual(meta["dataset_fingerprint"], config["sha256"])
                        self.assertEqual(ids[-1], eos)
                        stops = [index for index, value in enumerate(ids) if value == eos]
                        self.assertEqual(len(stops), meta["records"])
                        start = 0
                        for stop in stops:
                            observed[tuple(ids[start:stop])] += 1
                            start = stop + 1
                    finally:
                        tokens._mmap.close()  # Release the Windows file handle before temp cleanup.
                    pair.append(Path(config[key]).read_bytes())
                self.assertEqual(observed, expected)
                outputs.append(pair)
                with self.assertRaises(FileExistsError):
                    run(config)
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
