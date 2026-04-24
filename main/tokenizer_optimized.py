"""
An optimized Byte Pair Encoding (BPE) tokenizer implementation.
This is the same as the tokenizer_optimized.py in the `https://github.com/Siyuan-Harry/bpe-optimized-from-scratch` repo.
I include this into the project just to avoid dependency warning on `run_train_model.py`.
"""
import regex as re
from typing import Iterable, Iterator, Dict, List, Tuple
import json

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        self.vocab = vocab
        # Optimization 1: Convert merges into a rank dictionary to achieve O(1) query rule order
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        
        # Optimization 2: Pre-build the reverse vocabulary
        self.reverted_vocab = {v: k for k, v in self.vocab.items()}
        
        # Optimization 3: Introduce caching (key to speeding up BPE)
        # Store mapping from bytes -> list[int]
        self.cache: Dict[bytes, List[int]] = {}

        self.special_token_to_id: Dict[str, int] = {}
        if self.special_tokens:
            for tok in self.special_tokens:
                tok_bytes = tok.encode("utf-8")
                found_ids = [i for i, b in self.vocab.items() if b == tok_bytes]
                if found_ids:
                    self.special_token_to_id[tok] = found_ids[0]
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
            vocab_json = json.load(f)
        vocab = {int(k): v.encode("latin1") for k, v in vocab_json.items()}

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
        merges = [(p[0].encode("latin1"), p[1].encode("latin1")) for p in merges_json]

        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: bytes) -> List[int]:
        """对单个 pre-token 进行 BPE 合并，带缓存优化"""
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        word = [bytes([b]) for b in token_bytes]
        
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            best_pair = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))
            
            if best_pair not in self.ranks:
                break
                
            # 执行合并
            new_word = []
            i = 0
            p1, p2 = best_pair
            while i < len(word):
                if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                    new_word.append(p1 + p2)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            
        ids = [self.reverted_vocab[tok] for tok in word]
        self.cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        token_ids = []
        
        if self.specials_regex:
            segments = self.specials_regex.split(text)
        else:
            segments = [text]

        for seg in segments:
            if not seg: continue
            if seg in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[seg])
            else:
                for m in self.pat.finditer(seg):
                    pre_token_bytes = m.group(0).encode("utf-8")
                    token_ids.extend(self._bpe(pre_token_bytes))
        
        return token_ids

    def decode(self, ids: List[int]) -> str:
        byte_stream = b"".join(self.vocab[idx] for idx in ids)
        return byte_stream.decode('utf-8', errors="replace")