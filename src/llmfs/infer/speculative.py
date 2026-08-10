"""Speculative decoding, implemented by hand.

Autoregressive decoding is latency-bound by a serial dependency: token *n+1* cannot be
computed until *n* exists, so each token costs one full pass over the weights no matter
how little arithmetic it needs. The GPU is almost idle — a 124M model at batch 1 uses a
few percent of an H100's FLOPs while saturating its memory bandwidth.

Speculative decoding trades that spare compute for latency. A cheap *drafter* proposes
``k`` tokens; the target model scores all ``k+1`` positions in **one** forward pass,
which costs barely more than scoring one; and the proposals are accepted up to the
first disagreement. Every accepted token is a token the target model never had to
generate serially.

The invariant
-------------
**The output must be exactly what the target model would have produced alone.** That is
what makes this an optimisation rather than an approximation, and it is the property
``tests/test_speculative.py`` asserts — same prompt, same seed, token-for-token
identical against ordinary decoding. A speculative implementation that is merely
*close* is not fast, it is wrong.

Under greedy decoding the rule is simple: accept a draft token if it equals the
target's argmax at that position, stop at the first mismatch, and always take the
target's own token for the mismatched position. That last detail is what guarantees
progress — even a drafter that is wrong every single time still yields one token per
iteration, so the worst case is the speed of ordinary decoding plus the drafter's cost,
never a stall.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from ..model import GenerationConfig, Transformer


@dataclass
class SpecStats:
    """Where the speedup came from, or did not."""

    iterations: int = 0
    proposed: int = 0
    accepted: int = 0
    target_forwards: int = 0
    drafter_forwards: int = 0
    tokens_generated: int = 0
    accepted_per_iteration: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed tokens the target agreed with.

        The single number that decides whether speculation pays. Below roughly 1/k it
        cannot: the drafter's cost is not recovered."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_forward(self) -> float:
        """Speedup ceiling. Ordinary decoding is exactly 1.0 by definition."""
        return self.tokens_generated / self.target_forwards if self.target_forwards else 0.0

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "target_forwards": self.target_forwards,
            "drafter_forwards": self.drafter_forwards,
            "tokens_generated": self.tokens_generated,
            "tokens_per_target_forward": round(self.tokens_per_target_forward, 3),
        }


class Drafter(ABC):
    """Proposes continuations. Cheapness is the only requirement — a drafter that costs
    as much as the target model cannot win, however accurate it is."""

    name: str = "drafter"

    @abstractmethod
    def propose(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        """Return up to ``k`` proposed next tokens for the ``(1, T)`` sequence."""

    def reset(self) -> None:  # noqa: B027 - optional hook, not every drafter has state
        """Clear any per-sequence state. Stateless drafters need not override this."""

    @property
    def forwards(self) -> int:
        return 0


class ModelDrafter(Drafter):
    """A smaller (or less-trained) model of the same vocabulary.

    The classic arrangement. The draft model must share the target's tokenizer — it is
    proposing token ids, and an id means something different under another vocabulary.
    """

    def __init__(self, model: Transformer, name: str = "model") -> None:
        self.model = model.eval()
        self.name = name
        self._forwards = 0

    @torch.inference_mode()
    def propose(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        # Deliberately without a KV cache: with one, the drafter's cache would have to
        # be rolled back on every rejection, and getting that wrong silently corrupts
        # the proposals. This costs the drafter throughput and keeps it obviously
        # correct — the honest trade for a from-scratch implementation, and it is
        # noted in the results.
        out = []
        current = tokens
        for _ in range(k):
            logits = self.model(current[:, -self.model.cfg.block_size :]).logits[:, -1, :]
            self._forwards += 1
            nxt = logits.argmax(dim=-1, keepdim=True)
            out.append(nxt)
            current = torch.cat([current, nxt], dim=1)
        return torch.cat(out, dim=1) if out else tokens[:, :0]

    @property
    def forwards(self) -> int:
        return self._forwards


class PromptLookupDrafter(Drafter):
    """Draft by copying from the context itself — no model, no cost.

    Finds the most recent earlier occurrence of the last ``n`` tokens and proposes
    whatever followed it. Text is repetitive: quotations, names, code, list structure,
    and the boilerplate a small model falls into. Where it applies the acceptance rate
    is very high and the drafter is free, which makes it the best speedup-per-effort
    available. Where it does not, it proposes nothing and costs nothing.
    """

    name = "prompt-lookup"

    def __init__(self, max_ngram: int = 3, min_ngram: int = 1) -> None:
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram

    def propose(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        ids = tokens[0].tolist()
        # Longest match first: a longer context is a stronger predictor, so it is worth
        # trying before falling back to a shorter, more ambiguous one.
        for n in range(min(self.max_ngram, len(ids) - 1), self.min_ngram - 1, -1):
            suffix = ids[-n:]
            for start in range(len(ids) - n - 1, -1, -1):
                if ids[start : start + n] == suffix:
                    nxt = ids[start + n : start + n + k]
                    if nxt:
                        return torch.tensor([nxt], dtype=tokens.dtype, device=tokens.device)
        return tokens[:, :0]


@torch.inference_mode()
def speculative_generate(
    target: Transformer,
    drafter: Drafter,
    prompt: torch.Tensor,
    max_new_tokens: int = 128,
    k: int = 4,
    stats: SpecStats | None = None,
) -> tuple[torch.Tensor, SpecStats]:
    """Greedy speculative decoding. Output is identical to ``target.generate`` greedy.

    The target keeps a KV cache across iterations and is fed only the tokens it has not
    seen. Without that, each verification re-processes the whole prefix and the
    algorithm's advantage is spent on redundant attention — the measured effect was
    stark: 8.0 tokens per target forward pass still running at 0.66x wall-clock.

    Rejected drafts are undone with :meth:`KVCache.rewind_to`. Because the cache is
    preallocated, that is a move of the write offset rather than a reallocation, so a
    rejection costs essentially nothing.

    Args:
        target: the model whose output distribution must be preserved exactly.
        drafter: proposes candidates.
        prompt: ``(1, T)`` token ids. Batch 1 — batching speculation needs ragged
            accept lengths per row, which is a different algorithm.
        k: tokens proposed per iteration. Larger k wins more when the drafter is good
            and wastes more target compute when it is not.
    """
    if prompt.shape[0] != 1:
        raise ValueError(f"speculative decoding here is batch-1 only, got {prompt.shape[0]}")

    stats = stats or SpecStats()
    drafter.reset()
    tokens = prompt

    total = prompt.shape[1] + max_new_tokens
    if total > target.cfg.block_size:
        raise ValueError(
            f"prompt ({prompt.shape[1]}) + max_new_tokens ({max_new_tokens}) = {total} "
            f"exceeds block_size={target.cfg.block_size}"
        )
    # Sized for the whole run plus one iteration's worth of speculation, since the
    # verify pass writes k entries before any rejection is rolled back.
    cache = target.make_cache(1, max_seq_len=total + k + 1, device=prompt.device)

    produced = 0
    while produced < max_new_tokens:
        budget = max_new_tokens - produced
        draft = drafter.propose(tokens, min(k, budget))
        n_draft = draft.shape[1]

        # Everything the cache has not seen: tokens carried over from the previous
        # iteration's accepted correction, plus this iteration's draft.
        pending = tokens[:, cache.pos :]
        to_feed = torch.cat([pending, draft], dim=1) if n_draft else pending
        n_pending = pending.shape[1]
        settled = cache.pos + n_pending  # cache position covering only real tokens

        logits = target(to_feed, targets=to_feed, cache=cache).logits
        stats.target_forwards += 1
        stats.iterations += 1
        stats.proposed += n_draft

        # Position i of to_feed predicts to_feed[i+1]. The last n_draft+1 predictions
        # are the target's opinion of each draft token, plus its own next token.
        predictions = logits[0, -(n_draft + 1) :, :].argmax(dim=-1)

        n_accept = 0
        while n_accept < n_draft and predictions[n_accept] == draft[0, n_accept]:
            n_accept += 1

        # Accept the agreed prefix, then take the target's own token for the first
        # disagreement. This guarantees progress: a drafter that is always wrong still
        # yields one real token per iteration.
        accepted = draft[:, :n_accept]
        correction = predictions[n_accept].reshape(1, 1)
        new = torch.cat([accepted, correction], dim=1)

        take = min(new.shape[1], budget)
        tokens = torch.cat([tokens, new[:, :take]], dim=1)
        produced += take

        # Drop the rejected drafts from the cache. The correction token is deliberately
        # left out too — it becomes next iteration's `pending`, which keeps one code
        # path for both it and the prompt.
        cache.rewind_to(settled + min(n_accept, take))

        stats.accepted += n_accept
        stats.accepted_per_iteration.append(n_accept)
        stats.tokens_generated += take

    stats.drafter_forwards = drafter.forwards
    return tokens, stats


@torch.inference_mode()
def greedy_generate(
    target: Transformer, prompt: torch.Tensor, max_new_tokens: int = 128
) -> torch.Tensor:
    """Ordinary greedy decoding with a KV cache — the reference output and baseline."""
    return target.generate(
        prompt,
        GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0, top_k=None),
        use_cache=True,
    )
