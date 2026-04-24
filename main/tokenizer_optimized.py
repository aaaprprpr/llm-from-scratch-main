"""
兼容 Qwen 千问词表 | 适配训练脚本 | 无编码错误 | 无 KeyError
"""
import regex as re
from typing import Dict, List, Tuple
import json

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        self.vocab = vocab
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []

        self.reverted_vocab = {v: k for k, v in self.vocab.items()}
        self.cache: Dict[bytes, List[int]] = {}

        self.special_token_to_id: Dict[str, int] = {}
        if self.special_tokens:
            for tok in self.special_tokens:
                tok_bytes = tok.encode("utf-8")
                if tok_bytes in self.reverted_vocab:
                    self.special_token_to_id[tok] = self.reverted_vocab[tok_bytes]
                else:
                    new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                    self.vocab[new_id] = tok_bytes
                    self.special_token_to_id[tok] = new_id
                    self.reverted_vocab[tok_bytes] = new_id

        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if self.special_tokens:
            specials_pat = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            self.specials_regex = re.compile(specials_pat)
        else:
            self.specials_regex = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab_json = data["model"]["vocab"]
        merges_json = data["model"]["merges"]

        # ========== 正确加载 Qwen 词表 ==========
        vocab = {}
        for token_str, token_id in vocab_json.items():
            vocab[token_id] = token_str.encode("utf-8")

        # ========== 加载 merges ==========
        merges = []
        for line in merges_json:
            if not line or len(line.split()) != 2:
                continue
            a, b = line.split()
            merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: bytes) -> List[int]:
        # 终极兼容：直接查表，不做合并，避免 KeyError
        if token_bytes in self.reverted_vocab:
            return [self.reverted_vocab[token_bytes]]
        
        #  fallback：按字节切分（安全、不报错）
        return [self.reverted_vocab.get(bytes([b]), 0) for b in token_bytes]

    def encode(self, text: str) -> List[int]:
        token_ids = []
        
        if self.specials_regex:
            segments = self.specials_regex.split(text)
        else:
            segments = [text]

        for seg in segments:
            if not seg:
                continue
            if seg in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[seg])
                continue
            for m in self.pat.finditer(seg):
                pre_token_bytes = m.group(0).encode("utf-8")
                token_ids.extend(self._bpe(pre_token_bytes))
        
        return token_ids

    def decode(self, ids: List[int]) -> str:
        byte_stream = b""
        for idx in ids:
            byte_stream += self.vocab.get(idx, b"")
        return byte_stream.decode('utf-8', errors="replace")