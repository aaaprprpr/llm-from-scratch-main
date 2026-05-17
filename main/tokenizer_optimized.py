from typing import  List
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import torch
class Tokenizer:
    def __init__(self, file_path:str):
        if file_path.endswith(".json"):
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(file_path)
        self.special_token_to_id = self.tokenizer.get_added_vocab()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text,add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    
    def tokenize(self,text: str) -> List[str]:
        print(self.tokenizer.tokenize(text, add_special_tokens=False))
        return self.tokenizer.tokenize(text, add_special_tokens=False)
    
    def idx(self,text:str,device):
        return torch.tensor(self.encode(text),dtype=torch.long,device=device).unsqueeze(0)
    def text(self,idx,device):
        return self.decode(idx[0].tolist())