import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import StaticKVCache, Transformer
from pretrain.play_model import load_model as load_play_model
from pretrain.train_model import (
    get_batch,
    load_checkpoint,
    load_token_bin,
    save_checkpoint,
)

try:
    from hf.configuration_llm_from_scratch import LLMFromScratchConfig
    from hf.modeling_llm_from_scratch import LLMFromScratchForCausalLM

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def make_model(context_length=4096, num_layers=1):
    return Transformer(
        vocab_size=33,
        context_length=context_length,
        n_head=2,
        theta=10000.0,
        num_layers=num_layers,
        d_model=16,
        d_ff=32,
    )


class LongContextModelTest(unittest.TestCase):
    def test_2k_and_4k_forward(self):
        model = make_model().eval()
        with torch.no_grad():
            for sequence_length in (2048, 4096):
                input_ids = torch.randint(0, 33, (1, sequence_length))
                logits, cache = model(input_ids)
                self.assertEqual(
                    logits.shape,
                    (1, sequence_length, 33),
                )
                self.assertTrue(torch.isfinite(logits).all())
                self.assertIsNone(cache)

        with self.assertRaisesRegex(ValueError, "exceeds context_length"):
            model(torch.zeros(1, 4097, dtype=torch.long))

    def test_context_extension_keeps_state_dict_compatible(self):
        short_model = make_model(context_length=256)
        long_model = make_model(context_length=4096)

        self.assertEqual(
            set(short_model.state_dict()),
            set(long_model.state_dict()),
        )
        self.assertFalse(
            any("rope" in key for key in long_model.state_dict())
        )
        long_model.load_state_dict(short_model.state_dict(), strict=True)

        self.assertEqual(long_model.rope.cos_cached.shape, (4096, 4))
        cache_pointer = long_model.rope.cos_cached.data_ptr()
        with torch.no_grad():
            long_model(torch.randint(0, 33, (1, 2048)))
            long_model(torch.randint(0, 33, (1, 4096)))
        self.assertEqual(long_model.rope.cos_cached.data_ptr(), cache_pointer)
        self.assertFalse(hasattr(long_model.layers[0].attn, "rope"))

    def test_dtype_repair_does_not_expand_rope_capacity(self):
        model = make_model(context_length=64).bfloat16().eval()
        with torch.no_grad():
            model(torch.randint(0, 33, (1, 16)))

        self.assertEqual(model.rope.cos_cached.shape, (64, 4))
        self.assertEqual(model.rope.cos_cached.dtype, torch.float32)
        self.assertEqual(model.rope.inv_freq.dtype, torch.float32)

    def test_position_embeddings_are_cast_once_and_shared(self):
        model = make_model(context_length=64, num_layers=3).eval()
        position_tables = []

        def capture_position_embeddings(module, args, kwargs):
            position_tables.append(kwargs["position_embeddings"])

        hooks = [
            layer.attn.register_forward_pre_hook(
                capture_position_embeddings,
                with_kwargs=True,
            )
            for layer in model.layers
        ]
        try:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                model(torch.randint(0, 33, (1, 16)))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(len(position_tables), 3)
        first_cos, first_sin = position_tables[0]
        self.assertEqual(first_cos.dtype, torch.bfloat16)
        self.assertEqual(first_sin.dtype, torch.bfloat16)
        for cos, sin in position_tables[1:]:
            self.assertEqual(cos.data_ptr(), first_cos.data_ptr())
            self.assertEqual(sin.data_ptr(), first_sin.data_ptr())

    def test_cached_decode_matches_full_forward(self):
        torch.manual_seed(0)
        model = make_model(context_length=128, num_layers=2).eval()
        input_ids = torch.randint(0, 33, (2, 37))

        with torch.no_grad():
            full_logits, _ = model(input_ids)
            _, cache = model(input_ids[:, :19], use_cache=True)
            suffix_logits, _ = model(
                input_ids[:, 19:],
                past_key_values=cache,
                use_cache=True,
            )

            token_logits = []
            cache = None
            for position in range(input_ids.size(1)):
                logits, cache = model(
                    input_ids[:, position : position + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                token_logits.append(logits)

        torch.testing.assert_close(
            suffix_logits,
            full_logits[:, 19:],
            atol=2e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            torch.cat(token_logits, dim=1),
            full_logits,
            atol=2e-6,
            rtol=1e-5,
        )

    def test_static_cache_matches_full_forward_without_reallocation(self):
        class CatCounter(TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                if func is torch.cat:
                    self.calls += 1
                return func(*args, **(kwargs or {}))

        torch.manual_seed(3)
        model = make_model(context_length=128, num_layers=2).eval()
        input_ids = torch.randint(0, 33, (2, 37))
        cache = model.create_static_kv_cache(
            batch_size=input_ids.size(0),
            max_cache_length=input_ids.size(1),
        )

        with torch.no_grad():
            full_logits, _ = model(input_ids)
            first_logits, returned_cache = model(
                input_ids[:, :19],
                past_key_values=cache,
                use_cache=True,
            )
            pointers = [
                (key.data_ptr(), value.data_ptr())
                for key, value in zip(cache.key_cache, cache.value_cache)
            ]

            cat_counter = CatCounter()
            with cat_counter:
                second_logits, returned_cache = model(
                    input_ids[:, 19:20],
                    past_key_values=cache,
                    use_cache=True,
                )
            third_logits, returned_cache = model(
                input_ids[:, 20:],
                past_key_values=cache,
                use_cache=True,
            )

        self.assertIs(returned_cache, cache)
        self.assertEqual(cache.get_seq_length(), input_ids.size(1))
        self.assertEqual(cat_counter.calls, 0)
        self.assertEqual(
            pointers,
            [
                (key.data_ptr(), value.data_ptr())
                for key, value in zip(cache.key_cache, cache.value_cache)
            ],
        )
        torch.testing.assert_close(
            torch.cat((first_logits, second_logits, third_logits), dim=1),
            full_logits,
            atol=2e-6,
            rtol=1e-5,
        )

    def test_static_cache_capacity(self):
        model = make_model(context_length=32, num_layers=2).eval()
        cache = model.create_static_kv_cache(
            batch_size=1,
            max_cache_length=8,
        )

        with torch.no_grad():
            model(
                torch.tensor([[1, 2, 3]]),
                past_key_values=cache,
                use_cache=True,
            )

        with self.assertRaisesRegex(ValueError, "static capacity"):
            with torch.no_grad():
                model(
                    torch.tensor([[4, 5, 6, 7, 8, 9]]),
                    past_key_values=cache,
                    use_cache=True,
                )

    def test_checkpointing_preserves_loss_and_gradients(self):
        torch.manual_seed(1)
        plain_model = make_model(context_length=64, num_layers=2).train()
        checkpointed_model = make_model(
            context_length=64,
            num_layers=2,
        ).train()
        checkpointed_model.load_state_dict(plain_model.state_dict())
        checkpointed_model.gradient_checkpointing_enable()

        input_ids = torch.randint(0, 33, (1, 32))
        targets = torch.randint(0, 33, (1, 32))
        plain_loss = F.cross_entropy(
            plain_model(input_ids)[0].reshape(-1, 33),
            targets.reshape(-1),
        )
        checkpointed_loss = F.cross_entropy(
            checkpointed_model(input_ids)[0].reshape(-1, 33),
            targets.reshape(-1),
        )
        plain_loss.backward()
        checkpointed_loss.backward()

        torch.testing.assert_close(plain_loss, checkpointed_loss)
        for plain_parameter, checkpointed_parameter in zip(
            plain_model.parameters(),
            checkpointed_model.parameters(),
        ):
            torch.testing.assert_close(
                plain_parameter.grad,
                checkpointed_parameter.grad,
            )

        with self.assertRaisesRegex(ValueError, "incompatible"):
            checkpointed_model(input_ids, use_cache=True)

    def test_gradient_accumulation_matches_larger_batch(self):
        torch.manual_seed(2)
        large_batch_model = make_model(context_length=32).train()
        accumulated_model = make_model(context_length=32).train()
        accumulated_model.load_state_dict(large_batch_model.state_dict())

        input_ids = torch.randint(0, 33, (2, 16))
        targets = torch.randint(0, 33, (2, 16))
        large_batch_loss = F.cross_entropy(
            large_batch_model(input_ids)[0].reshape(-1, 33),
            targets.reshape(-1),
        )
        large_batch_loss.backward()

        for micro_batch in range(2):
            micro_loss = F.cross_entropy(
                accumulated_model(
                    input_ids[micro_batch : micro_batch + 1]
                )[0].reshape(-1, 33),
                targets[micro_batch : micro_batch + 1].reshape(-1),
            )
            (micro_loss / 2).backward()

        for large_parameter, accumulated_parameter in zip(
            large_batch_model.parameters(),
            accumulated_model.parameters(),
        ):
            torch.testing.assert_close(
                large_parameter.grad,
                accumulated_parameter.grad,
                atol=2e-6,
                rtol=1e-5,
            )

    def test_position_validation_and_broadcasting(self):
        model = make_model(context_length=64).eval()
        input_ids = torch.randint(0, 33, (2, 12))
        positions = torch.arange(12).expand(2, -1)

        with torch.no_grad():
            default_logits, _ = model(input_ids)
            explicit_logits, _ = model(
                input_ids,
                token_positions=positions,
            )
        torch.testing.assert_close(default_logits, explicit_logits)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            model(input_ids, token_positions=positions - 1)
        with self.assertRaisesRegex(ValueError, "position range"):
            model(input_ids, token_positions=positions + 64)
        with self.assertRaisesRegex(ValueError, "batch dimension"):
            model(
                input_ids,
                token_positions=torch.arange(12).expand(3, -1),
            )


class PretrainDataAndCheckpointTest(unittest.TestCase):
    def test_get_batch_reads_one_shifted_window(self):
        data = np.arange(24, dtype=np.uint16)
        x, y, position = get_batch(
            data,
            batch_size=2,
            context_length=4,
            device="cpu",
            position=0,
        )

        torch.testing.assert_close(
            x,
            torch.tensor(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ]
            ),
        )
        torch.testing.assert_close(
            y,
            torch.tensor(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ),
        )
        self.assertEqual(position, 8)
        self.assertEqual(x.dtype, torch.int64)
        self.assertEqual(y.dtype, torch.int64)
        self.assertTrue(x.is_contiguous())
        self.assertTrue(y.is_contiguous())

    def test_get_batch_wraps_before_an_incomplete_batch(self):
        data = np.arange(24, dtype=np.uint16)
        x, y, position = get_batch(
            data,
            batch_size=2,
            context_length=4,
            device="cpu",
            position=16,
        )

        torch.testing.assert_close(
            x,
            torch.tensor(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ]
            ),
        )
        torch.testing.assert_close(
            y,
            torch.tensor(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ),
        )
        self.assertEqual(position, 8)

    def test_load_token_bin_uses_metadata_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_path = Path(directory) / "tokens.bin"
            expected = np.array([0, 65_536, 100_000], dtype=np.uint32)
            expected.tofile(bin_path)
            bin_path.with_suffix(".bin.meta.json").write_text(
                '{"dtype": "uint32"}',
                encoding="utf-8",
            )

            actual = load_token_bin(bin_path)

        self.assertEqual(actual.dtype, np.dtype("uint32"))
        np.testing.assert_array_equal(actual, expected)

    def test_checkpoint_round_trip_includes_long_context_metadata(self):
        model = make_model(context_length=4096)
        optimizer = torch.optim.AdamW(model.parameters())
        model_args = {
            "vocab_size": 33,
            "context_length": 4096,
            "n_head": 2,
            "theta": 10000.0,
            "num_layers": 1,
            "d_model": 16,
            "d_ff": 32,
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                model,
                optimizer,
                iteration=7,
                out=checkpoint_path,
                train_position=123,
                tokens_seen=28672,
                model_args=model_args,
                config={"runtime": {"sequence_length": 2048}},
            )

            restored_model = make_model(context_length=4096)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters())
            iteration, position, tokens_seen = load_checkpoint(
                checkpoint_path,
                restored_model,
                restored_optimizer,
            )

        self.assertEqual(iteration, 7)
        self.assertEqual(position, 123)
        self.assertEqual(tokens_seen, 28672)
        for expected, actual in zip(
            model.parameters(),
            restored_model.parameters(),
        ):
            torch.testing.assert_close(expected, actual)

    def test_play_script_uses_checkpoint_model_args(self):
        model_args = {
            "vocab_size": 33,
            "context_length": 64,
            "n_head": 2,
            "theta": 10000.0,
            "num_layers": 1,
            "d_model": 16,
            "d_ff": 32,
        }
        model = Transformer(**model_args)

        class FallbackConfig:
            @staticmethod
            def require(section):
                raise AssertionError(
                    f"checkpoint model_args should be used, requested {section}"
                )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_args": model_args,
                },
                checkpoint_path,
            )
            restored_model, restored_args = load_play_model(
                FallbackConfig(),
                checkpoint_path,
                torch.device("cpu"),
            )

        self.assertEqual(restored_args, model_args)
        self.assertEqual(restored_model.context_length, 64)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers is not installed")
class HuggingFaceCompatibilityTest(unittest.TestCase):
    def test_generation_uses_static_cache(self):
        config = LLMFromScratchConfig(
            vocab_size=33,
            context_length=64,
            n_head=2,
            num_layers=1,
            d_model=16,
            d_ff=32,
            theta=10000.0,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=None,
            pad_token_id=0,
        )
        model = LLMFromScratchForCausalLM(config).eval()
        with torch.no_grad():
            output = model.generate(
                torch.tensor([[2, 3, 4]]),
                max_new_tokens=4,
                do_sample=False,
                return_dict_in_generate=True,
            )

        self.assertEqual(output.sequences.shape, (1, 7))
        self.assertIsInstance(output.past_key_values, StaticKVCache)
        self.assertEqual(output.past_key_values.get_seq_length(), 6)


if __name__ == "__main__":
    unittest.main()
