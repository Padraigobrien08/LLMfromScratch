/**
 * The tests carrying `@pytest.mark.showcase`, collected by pytest itself.
 *
 * Renaming or deleting one of them changes this file, so the page cannot go on
 * advertising a guarantee the suite no longer provides. `cases` is how many
 * parametrised runs stand behind the claim.
 *
 * Do not hand-edit — regenerate with `llmfs-export-web`. `tests/test_web_export.py`
 * asserts this file is still what the generator emits, so a stale copy fails CI
 * rather than shipping.
 */
export const TEST_SHOWCASE = [
  {
    "file": "tests/test_attention.py",
    "name": "test_build_causal_mask_is_bottom_right_aligned",
    "pins": "that the causal mask is bottom-right aligned, not top-left",
    "why": "torch's is_causal=True is top-left aligned, which is silently wrong whenever the query block is shorter than the key sequence \u2014 that is, on every single decode step. It would let the newest token see only the oldest keys.",
    "cases": 1
  },
  {
    "file": "tests/test_attention.py",
    "name": "test_causality",
    "pins": "that perturbing token t leaves every position < t bitwise unchanged",
    "why": "An off-by-one in the mask makes the loss look *better*, because the model is peeking at the answer. Nothing downstream of a loss curve can catch that.",
    "cases": 10
  },
  {
    "file": "tests/test_attention.py",
    "name": "test_eager_matches_sdpa",
    "pins": "that the weight-exporting eager path computes what the fused kernel computes",
    "why": "The attention visualizer reads its maps from the eager path. If the two ever disagreed it would be drawing weights the model never used \u2014 and a picture that has quietly drifted from the code is worse than no picture, because nothing about it looks wrong.",
    "cases": 3
  },
  {
    "file": "tests/test_attention.py",
    "name": "test_gqa_reduces_to_mha",
    "pins": "that with n_kv_head == n_head, grouped-query attention is numerically identical to plain multi-head attention",
    "why": "Same weights, same inputs, so the grouping code is the only difference. It turns the repeat/interleave logic \u2014 the easiest thing in GQA to get subtly wrong \u2014 into a difference that has to be exactly zero.",
    "cases": 1
  },
  {
    "file": "tests/test_config.py",
    "name": "test_ablation_arms_differ_from_their_baseline_in_one_axis_only",
    "pins": "that every ablation arm differs from the shared baseline in its own named axis and nothing else",
    "why": "The discipline the entire ablation study rests on. An arm that drifted would measure something other than the thing named on the tin, and the resulting table would read as authoritative while being about nothing in particular.",
    "cases": 1
  },
  {
    "file": "tests/test_kv_cache.py",
    "name": "test_decode_step_passes_no_attn_mask",
    "pins": "that a single-token decode step reaches SDPA with attn_mask=None",
    "why": "The one test here that checks *speed* rather than an answer. Passing a mask disqualifies SDPA from its fused kernels and drops it onto the math backend; when q_len == 1 that mask is all-True and therefore pure cost. Its absence let a 30% regression sit behind a green suite for weeks.",
    "cases": 1
  },
  {
    "file": "tests/test_kv_cache.py",
    "name": "test_incremental_decode_matches_full_forward",
    "pins": "that token-at-a-time decoding with a cache reproduces a full forward pass at every position",
    "why": "Training never exercises the cache, so nothing else in the suite would catch a stale offset or a double-rotated key. The model would train perfectly and generate subtly wrong text.",
    "cases": 10
  },
  {
    "file": "tests/test_kv_cache.py",
    "name": "test_multi_token_verify_step_still_masks",
    "pins": "that a multi-token block against a filled cache still gets a real mask, and its *interior* queries agree with a full forward pass",
    "why": "The speculative-verification shape. The q_len == 1 fast path must not swallow it \u2014 and checking only the final position would pass regardless, because that one query is correct even with no mask at all.",
    "cases": 1
  },
  {
    "file": "tests/test_model.py",
    "name": "test_tying_couples_gradients",
    "pins": "that both roles of a tied embedding matrix contribute to its one gradient",
    "why": "Weight tying is easy to implement as a copy rather than a share. The model then trains, converges, and quietly optimises the output head against an input table that never receives its half of the signal.",
    "cases": 1
  },
  {
    "file": "tests/test_quant.py",
    "name": "test_quantizing_a_tied_head_is_refused",
    "pins": "that quantizing a tied lm_head is refused rather than silently done",
    "why": "With tied embeddings the head *is* the token embedding, so quantizing it stores a compressed copy while the original fp32 tensor stays. Measured on the 124M model that makes it bigger: 196 MiB becomes 217. Refusing beats reporting a compression ratio worse than doing nothing.",
    "cases": 1
  },
  {
    "file": "tests/test_rope.py",
    "name": "test_relative_position_property",
    "pins": "that <R(q, m), R(k, n)> depends only on m - n, to 1e-6",
    "why": "It is the entire point of rotary embeddings and no shape check implies it. A rotation applied to the wrong axis still produces the right-shaped tensor and a model that trains \u2014 just one with no usable notion of distance.",
    "cases": 1
  },
  {
    "file": "tests/test_speculative.py",
    "name": "test_output_is_identical_to_greedy_decoding",
    "pins": "that speculative decoding reproduces greedy decoding token for token",
    "why": "The whole contract. An implementation that were merely *close* would not be a faster decoder \u2014 it would be a different model, and every benchmark measuring it would be measuring the wrong thing.",
    "cases": 4
  }
] as const;
