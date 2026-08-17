"""Speculative decoding.

One test matters more than all the others: **the output must be identical to ordinary
decoding**. Speculative decoding is an optimisation, so any difference in what it
produces is a bug, not a tradeoff. A version that is merely close is not a faster
decoder — it is a different, worse model with extra steps.
"""

from __future__ import annotations

import pytest
import torch

from conftest import tiny_model
from llmfs.infer.speculative import (
    Drafter,
    ModelDrafter,
    PromptLookupDrafter,
    SpecStats,
    greedy_generate,
    speculative_generate,
)


class NullDrafter(Drafter):
    """Proposes nothing. The floor: speculation must still make progress."""

    name = "null"

    def propose(self, tokens, k):
        return tokens[:, :0]


class WrongDrafter(Drafter):
    """Always proposes a token the target will reject.

    The adversarial case. Every proposal is wasted, so this must still terminate and
    still produce the target's own output — just with no speedup.
    """

    name = "wrong"

    def __init__(self, bad_token: int = 0) -> None:
        self.bad_token = bad_token

    def propose(self, tokens, k):
        return torch.full((1, k), self.bad_token, dtype=tokens.dtype, device=tokens.device)


class OracleDrafter(Drafter):
    """Proposes exactly what the target would produce. The ceiling."""

    name = "oracle"

    def __init__(self, model):
        self.model = model.eval()

    @torch.inference_mode()
    def propose(self, tokens, k):
        out = []
        current = tokens
        for _ in range(k):
            nxt = self.model(current).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out.append(nxt)
            current = torch.cat([current, nxt], dim=1)
        return torch.cat(out, dim=1) if out else tokens[:, :0]


@pytest.fixture
def model():
    return tiny_model(vocab_size=97, n_layer=2, n_head=4, n_embd=64, block_size=128)


@pytest.fixture
def prompt():
    torch.manual_seed(0)
    return torch.randint(0, 97, (1, 8))


# ------------------------------------------------------- the losslessness invariant


@pytest.mark.showcase(
    pins="that speculative decoding reproduces greedy decoding token for token",
    why="The whole contract. An implementation that merely came *close* would not be a "
    "faster decoder — it would be a different model, and every benchmark measuring "
    "it would be measuring the wrong thing.",
)
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_output_is_identical_to_greedy_decoding(model, prompt, k) -> None:
    """The whole contract. Any difference here is a correctness bug."""
    reference = greedy_generate(model, prompt, max_new_tokens=24)
    out, _ = speculative_generate(model, OracleDrafter(model), prompt, max_new_tokens=24, k=k)
    assert torch.equal(out, reference), f"k={k} diverged from greedy decoding"


@pytest.mark.parametrize(
    "drafter_factory",
    [
        lambda m: NullDrafter(),
        lambda m: WrongDrafter(),
        lambda m: OracleDrafter(m),
        lambda m: PromptLookupDrafter(),
    ],
    ids=["null", "wrong", "oracle", "prompt-lookup"],
)
def test_every_drafter_gives_the_same_output(model, prompt, drafter_factory) -> None:
    """Drafter quality changes the speed, never the result.

    A useless drafter, an adversarial one and a perfect one must all produce exactly
    what the target model would have produced alone.
    """
    reference = greedy_generate(model, prompt, max_new_tokens=20)
    out, _ = speculative_generate(model, drafter_factory(model), prompt, max_new_tokens=20, k=4)
    assert torch.equal(out, reference)


def test_a_useless_drafter_still_makes_progress(model, prompt) -> None:
    """Zero accepted tokens must still yield one real token per iteration.

    This is why the target's own token is appended at the rejection point. Without it,
    a drafter that is always wrong would stall forever.
    """
    out, stats = speculative_generate(model, WrongDrafter(), prompt, max_new_tokens=12, k=4)
    assert out.shape[1] == prompt.shape[1] + 12
    assert stats.accepted == 0
    assert stats.tokens_per_target_forward == pytest.approx(1.0, abs=0.01)


# ------------------------------------------------------------------------- length


@pytest.mark.parametrize("n", [1, 5, 12, 33])
def test_generates_exactly_the_requested_number_of_tokens(model, prompt, n) -> None:
    """Accepting a run of tokens must not overshoot the budget."""
    out, stats = speculative_generate(model, OracleDrafter(model), prompt, max_new_tokens=n, k=8)
    assert out.shape[1] == prompt.shape[1] + n
    assert stats.tokens_generated == n


# -------------------------------------------------------------------------- stats


def test_oracle_drafter_accepts_everything(model, prompt) -> None:
    """The ceiling: with a perfect drafter, k+1 tokens come out of each forward pass."""
    _, stats = speculative_generate(model, OracleDrafter(model), prompt, max_new_tokens=20, k=4)
    assert stats.acceptance_rate == pytest.approx(1.0)
    # Each iteration yields the 4 accepted drafts plus the target's own next token.
    assert stats.tokens_per_target_forward > 4.0


def test_stats_account_for_every_token(model, prompt) -> None:
    _, stats = speculative_generate(model, WrongDrafter(), prompt, max_new_tokens=10, k=3)
    assert stats.tokens_generated == 10
    assert stats.proposed == sum(min(3, 10 - i) for i in range(0, 10))  # k per iteration
    assert stats.iterations == len(stats.accepted_per_iteration)
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_stats_serialise() -> None:
    s = SpecStats(iterations=2, proposed=8, accepted=6, target_forwards=2, tokens_generated=8)
    d = s.to_dict()
    assert d["acceptance_rate"] == 0.75
    assert d["tokens_per_target_forward"] == 4.0


# ---------------------------------------------------------------- prompt lookup


def test_prompt_lookup_copies_a_repeated_continuation() -> None:
    """The drafter's whole premise: text repeats, so the context predicts itself."""
    drafter = PromptLookupDrafter(max_ngram=3)
    # "1 2 3 4 5" appears earlier; after the trailing "1 2 3" it should propose 4, 5.
    tokens = torch.tensor([[1, 2, 3, 4, 5, 9, 9, 1, 2, 3]])
    proposal = drafter.propose(tokens, k=2)
    assert proposal.tolist() == [[4, 5]]


def test_prompt_lookup_proposes_nothing_when_there_is_no_match() -> None:
    """No match must cost nothing — an empty proposal, not a guess."""
    drafter = PromptLookupDrafter(max_ngram=3, min_ngram=2)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    assert drafter.propose(tokens, k=3).shape[1] == 0


def test_prompt_lookup_is_free(model, prompt) -> None:
    """It runs no model, so it adds no forward passes."""
    _, stats = speculative_generate(model, PromptLookupDrafter(), prompt, max_new_tokens=16, k=4)
    assert stats.drafter_forwards == 0


# --------------------------------------------------------------------- interfaces


def test_model_drafter_counts_its_own_cost(model, prompt) -> None:
    """A drafter that runs a model must report its forwards, or the speedup accounting
    silently ignores the cost that decides whether speculation pays."""
    small = tiny_model(vocab_size=97, n_layer=1, n_head=4, n_embd=64, block_size=128)
    drafter = ModelDrafter(small)
    _, stats = speculative_generate(model, drafter, prompt, max_new_tokens=12, k=3)
    assert stats.drafter_forwards > 0
    assert stats.target_forwards < stats.drafter_forwards


def test_a_reused_drafter_reports_this_run_not_every_run(model, prompt) -> None:
    """The counter must be per run, because the benchmark reuses one drafter for all of them.

    `ModelDrafter` incremented a counter and never cleared it — `Drafter.reset()` is a
    no-op hook and this class did not override it — so `benchmark.py`, which builds one
    drafter and sweeps it across every prompt and every k, recorded a running total. Two
    identical runs reported 13 forwards and then 26, and every model-draft row after the
    first in `results/speculative-cuda.json` carries the sum of the rows above it.
    """
    small = tiny_model(vocab_size=97, n_layer=1, n_head=4, n_embd=64, block_size=128)
    drafter = ModelDrafter(small)

    counts = [
        speculative_generate(model, drafter, prompt, max_new_tokens=12, k=3)[1].drafter_forwards
        for _ in range(3)
    ]
    assert counts[0] > 0
    assert counts == [counts[0]] * 3, f"identical runs reported {counts}"


def test_batched_prompts_are_rejected(model) -> None:
    """Ragged accept lengths per row make batched speculation a different algorithm;
    failing loudly beats silently decoding only row 0."""
    with pytest.raises(ValueError, match="batch-1"):
        speculative_generate(model, NullDrafter(), torch.zeros(2, 4, dtype=torch.long))
