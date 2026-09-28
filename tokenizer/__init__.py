"""Tokenizer loading and BPE training tools."""

__all__ = ["Tokenizer"]

# Keep corpus sampling free of transformers/torch imports until needed.
def __getattr__(name):
    if name == "Tokenizer":
        from .runtime import Tokenizer
        return Tokenizer
    raise AttributeError(name)
