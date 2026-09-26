"""Expose the wrapper moved under tokenize/ without shadowing stdlib tokenize."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_spec = spec_from_file_location("_project_tokenizer", Path(__file__).resolve().parent / "tokenize" / "tokenizer.py")
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load tokenize/tokenizer.py")
_module = module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
Tokenizer = _module.Tokenizer

__all__ = ["Tokenizer"]
