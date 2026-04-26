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

        # ✅ 支持中文的正则
        self.pat = re.compile(r"(\s+|\S+)")
        
        if self.special_tokens:
            specials_pat = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            self.specials_regex = re.compile(specials_pat)
        else:
            self.specials_regex = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None):
        # ==================== Qwen 官方标准字节映射 ====================
        def bytes_to_unicode():
            bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
            cs = bs[:]
            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1
            cs = [chr(n) for n in cs]
            return dict(zip(bs, cs))
        
        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab_json = data["model"]["vocab"]
        merges_json = data["model"]["merges"]

        # ✅ 正确加载 vocab
        vocab = {}
        for token_str, token_id in vocab_json.items():
            try:
                token_bytes = bytes([byte_decoder[c] for c in token_str])
            except:
                token_bytes = token_str.encode("utf-8")
            vocab[token_id] = token_bytes

        # ✅ 正确加载 merges
        merges = []
        for line in merges_json:
            if not line or line.startswith("#"):
                continue
            a, b = line.split(maxsplit=1)
            try:
                a_bytes = bytes([byte_decoder[c] for c in a])
                b_bytes = bytes([byte_decoder[c] for c in b])
            except:
                a_bytes = a.encode("utf-8")
                b_bytes = b.encode("utf-8")
            merges.append((a_bytes, b_bytes))

        return cls(vocab, merges, special_tokens)

    # ✅ 真正的 BPE 算法，没有被阉割
    def _bpe(self, token_bytes: bytes) -> List[int]:
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        word = [bytes([b]) for b in token_bytes]
        
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            best_pair = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))
            
            if best_pair not in self.ranks:
                break
                
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
        byte_stream = b"".join(self.vocab[idx] for idx in ids)
        return byte_stream.decode('utf-8', errors="replace")