"""Tokenizer loading.

Two backends behind one interface:

``gpt2``
    The original GPT-2 BPE via ``tiktoken``. Required for the reproduction — a
    validation loss is only comparable to a published one if the vocabulary that
    produced it is identical, since loss is per-token and a different tokenizer
    changes what a token is.
``file:<path>``
    A HuggingFace ``tokenizer.json``, including the one already in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol


class TokenizerBackend(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass
class Tokenizer:
    backend: TokenizerBackend
    vocab_size: int
    eot_token: int
    name: str

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        ids = self.backend.encode(text)
        return [self.eot_token, *ids] if add_eot else ids

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode(list(ids))


class _TiktokenBackend:
    def __init__(self, name: str = "gpt2") -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "the gpt2 tokenizer needs tiktoken — install it with `pip install tiktoken`"
            ) from exc
        self._enc = tiktoken.get_encoding(name)

    def encode(self, text: str) -> list[int]:
        # GPT-2's BPE has no special tokens in ordinary text; allowing none means a
        # document that literally contains "<|endoftext|>" raises instead of silently
        # injecting a document boundary.
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token


class _HFTokenizersBackend:
    def __init__(self, path: str | Path) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"tokenizer file not found: {path}")
        self._tok = HFTokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eot_token(self) -> int:
        for candidate in ("<|endoftext|>", "</s>", "[SEP]"):
            tid = self._tok.token_to_id(candidate)
            if tid is not None:
                return tid
        # No explicit document separator: fall back to id 0 and let the caller decide
        # whether document boundaries matter for their corpus.
        return 0


@lru_cache(maxsize=4)
def load_tokenizer(spec: str = "gpt2") -> Tokenizer:
    """Load a tokenizer from a spec string.

    Args:
        spec: ``gpt2``, or ``file:<path-to-tokenizer.json>``.
    """
    if spec.startswith("file:"):
        backend = _HFTokenizersBackend(spec[len("file:") :])
        return Tokenizer(
            backend=backend,
            vocab_size=backend.vocab_size,
            eot_token=backend.eot_token,
            name=spec,
        )
    if spec == "gpt2":
        backend = _TiktokenBackend("gpt2")
        return Tokenizer(
            backend=backend,
            vocab_size=backend.vocab_size,
            eot_token=backend.eot_token,
            name="gpt2",
        )
    raise ValueError(f"unknown tokenizer spec {spec!r}; expected 'gpt2' or 'file:<path>'")
