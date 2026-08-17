"""The training loop.

Deliberate choices worth stating, because they are the ones that make reported
numbers trustworthy:

* **Effective batch size is fixed in tokens, not in sequences.** ``tokens_per_step``
  is held at GPT-2's 524,288 and gradient accumulation is derived from it, so the
  optimisation trajectory is identical whether the run has one GPU or eight. A
  scaling experiment then measures throughput and nothing else.
* **Loss is normalised by accumulation steps.** Omitting this makes the gradient
  ``grad_accum`` times too large, which shows up as a learning rate that mysteriously
  needs retuning whenever the micro-batch changes.
* **Throughput and MFU are logged from the start.** They cost nothing to record and
  are the numbers that tell you whether a run is compute-bound or stalled on data.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import Config
from ..data.loader import ShardDataLoader, read_meta
from ..model import Transformer
from ..utils.device import autocast_context, enable_tf32, peak_flops, resolve_dtype
from ..utils.seed import load_rng_state, seed_everything
from .checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    unwrap_model,
)
from .distributed import all_reduce_mean, resolve_device, setup_distributed
from .optim import build_optimizer, lr_at_step, set_lr


class TrainingDiverged(RuntimeError):
    """Raised when the loss or gradient norm stops being finite.

    A distinct exception type so an ablation sweep can tell "this arm's learning
    rate was too high" — a result worth reporting — from "the code is broken".
    """


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = math.inf
    tokens_seen: int = 0
    last_eval_step: int = -1
    history: list[dict[str, Any]] = field(default_factory=list)


class MetricsLogger:
    """Writes metrics to JSONL, and to TensorBoard/W&B when enabled.

    The JSONL file is the source of truth for the ablation plots: it survives the
    process, needs no server, and can be diffed between runs.
    """

    def __init__(self, cfg: Config, run_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.run_dir = run_dir
        self._writer = None
        self._wandb = None
        if not enabled:
            return

        run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (run_dir / "metrics.jsonl").open("a")

        if cfg.log.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(run_dir / "tb"))
            except ImportError:
                print("[warn] tensorboard not installed; skipping (pip install -e '.[train]')")

        if cfg.log.wandb:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=cfg.log.wandb_project,
                    entity=cfg.log.wandb_entity,
                    name=cfg.log.run_name,
                    config=cfg.to_dict(),
                    dir=str(run_dir),
                )
            except ImportError:
                print("[warn] wandb not installed; skipping (pip install -e '.[train]')")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {"step": step, **metrics}
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        if self._writer is not None:
            for key, value in metrics.items():
                if isinstance(value, int | float):
                    self._writer.add_scalar(key, value, step)
        if self._wandb is not None:
            self._wandb.log(record, step=step)

    def close(self) -> None:
        if not self.enabled:
            return
        self._jsonl.close()
        if self._writer is not None:
            self._writer.close()
        if self._wandb is not None:
            self._wandb.finish()


class Trainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dist = setup_distributed()
        self.device = resolve_device(self.dist, cfg.runtime.device)
        self.dtype = resolve_dtype(cfg.runtime.dtype, self.device)

        # Offset the seed per rank so ranks do not share a dropout mask, while the
        # run as a whole stays reproducible from a single number.
        seed_everything(cfg.runtime.seed + self.dist.rank, cfg.runtime.deterministic)
        if cfg.runtime.tf32 and self.device.type == "cuda":
            enable_tf32()

        self.run_dir = Path(cfg.log.out_dir) / cfg.log.run_name
        self.grad_accum_steps = cfg.grad_accum_steps(self.dist.world_size)

        self.model = self._build_model()
        self.optimizer = build_optimizer(unwrap_model(self.model), cfg.optim, self.device)
        self.train_loader, self.val_loader = self._build_loaders()
        self.state = TrainState()
        self.diverged = False
        self.logger = MetricsLogger(cfg, self.run_dir, enabled=self.dist.is_main)

        if cfg.train.resume:
            self._resume(cfg.train.resume)

        if self.dist.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            cfg.save(self.run_dir / "config.yaml")
            self._print_banner()

    # ------------------------------------------------------------------- setup

    def _build_model(self) -> torch.nn.Module:
        cfg = self.cfg
        model = Transformer(cfg.model).to(self.device)

        if cfg.runtime.grad_checkpointing:
            self._enable_grad_checkpointing(model)

        if cfg.runtime.compile:
            model = torch.compile(model)

        if self.dist.enabled:
            from torch.nn.parallel import DistributedDataParallel

            device_ids = [self.dist.local_rank] if self.device.type == "cuda" else None
            model = DistributedDataParallel(model, device_ids=device_ids)

        return model

    @staticmethod
    def _enable_grad_checkpointing(model: Transformer) -> None:
        """Recompute each block's activations during the backward pass.

        Wrapping per block rather than per layer-internal is the right granularity:
        the block boundary is where the stored activation is a single (B, T, C)
        tensor, so the memory saved per recomputation is maximal.
        """
        from torch.utils.checkpoint import checkpoint

        for block in model.blocks:
            original_forward = block.forward

            def checkpointed(*args, _fn=original_forward, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_fn, *args, use_reentrant=False, **kwargs)
                return _fn(*args, **kwargs)

            block.forward = checkpointed  # type: ignore[method-assign]

    def _build_loaders(self) -> tuple[ShardDataLoader, ShardDataLoader]:
        cfg = self.cfg
        common = dict(
            data_dir=cfg.data.data_dir,
            micro_batch_size=cfg.data.micro_batch_size,
            block_size=cfg.data.block_size,
            rank=self.dist.rank,
            world_size=self.dist.world_size,
            device=self.device,
        )
        train = ShardDataLoader(split="train", **common)  # type: ignore[arg-type]
        val = ShardDataLoader(split="val", **common)  # type: ignore[arg-type]

        # A tokenizer mismatch between the data and the config is otherwise
        # invisible: training runs fine and simply never converges properly.
        meta = read_meta(cfg.data.data_dir)
        if meta.get("tokenizer") != cfg.data.tokenizer:
            raise ValueError(
                f"data in {cfg.data.data_dir} was built with tokenizer "
                f"{meta.get('tokenizer')!r} but the config asks for {cfg.data.tokenizer!r}"
            )
        if meta.get("vocab_size", 0) > cfg.model.vocab_size:
            raise ValueError(
                f"tokenizer vocab ({meta['vocab_size']}) exceeds model.vocab_size "
                f"({cfg.model.vocab_size}); the model cannot represent every token"
            )
        return train, val

    def _print_banner(self) -> None:
        model = unwrap_model(self.model)
        params = model.num_params()
        budget = self.cfg.train.max_steps * self.cfg.train.tokens_per_step
        budget_str = f"{budget / 1e9:.2f}B" if budget >= 1e9 else f"{budget / 1e6:.1f}M"
        print(
            f"\n{'=' * 68}\n"
            f"run           {self.cfg.log.run_name}\n"
            f"model         {params / 1e6:.1f}M params "
            f"({model.num_params(non_embedding=True) / 1e6:.1f}M non-embedding)\n"
            f"architecture  norm={self.cfg.model.norm} pos={self.cfg.model.pos_emb} "
            f"mlp={self.cfg.model.mlp} kv_heads={self.cfg.model.n_kv_head} "
            f"tied={self.cfg.model.tie_embeddings}\n"
            f"device        {self.device} dtype={self.dtype} "
            f"compile={self.cfg.runtime.compile} world_size={self.dist.world_size}\n"
            f"batch         {self.cfg.train.tokens_per_step:,} tokens/step = "
            f"{self.cfg.data.micro_batch_size} seq x {self.cfg.data.block_size} tok "
            f"x {self.grad_accum_steps} accum x {self.dist.world_size} rank(s)\n"
            f"schedule      {self.cfg.optim.schedule} lr={self.cfg.optim.lr:g} "
            f"warmup={self.cfg.optim.warmup_steps} steps={self.cfg.train.max_steps:,}\n"
            f"data          {self.train_loader.total_tokens:,} train / "
            f"{self.val_loader.total_tokens:,} val tokens\n"
            f"budget        {budget_str} tokens\n{'=' * 68}\n"
        )

    # ---------------------------------------------------------------- resuming

    def _resume(self, resume: str) -> None:
        path = find_latest_checkpoint(self.run_dir) if resume == "auto" else Path(resume)
        if path is None or not Path(path).exists():
            if resume == "auto":
                print("[resume] no checkpoint found; starting fresh")
                return
            raise FileNotFoundError(f"checkpoint not found: {resume}")

        ckpt = load_checkpoint(path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.state.step = ckpt["step"]
        self.state.best_val_loss = ckpt.get("metrics", {}).get("best_val_loss", math.inf)
        self.state.tokens_seen = self.state.step * self.cfg.train.tokens_per_step

        # The loader carries no state of its own; the step alone determines it.
        self.train_loader.set_step(self.state.step, self.grad_accum_steps)

        # Pick the random stream back up where it stopped rather than reseeding from the
        # config, which would replay step 0's dropout masks. Checkpoints written before
        # this was recorded have no `rng` key and keep the old, reseeded behaviour.
        restored = load_rng_state(ckpt.get("rng"))
        note = "" if restored else "  (no RNG state in this checkpoint; reseeded)"
        print(f"[resume] {path} at step {self.state.step:,}{note}")

    # ---------------------------------------------------------------- training

    def train(self) -> TrainState:
        cfg = self.cfg
        model = self.model
        model.train()

        t0 = time.perf_counter()
        try:
            while self.state.step < cfg.train.max_steps:
                step = self.state.step
                lr = lr_at_step(step, cfg.optim, cfg.train.max_steps)
                set_lr(self.optimizer, lr)

                loss_accum = self._accumulate_gradients()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                self._assert_finite(loss_accum, float(grad_norm))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.state.step += 1
                self.state.tokens_seen += cfg.train.tokens_per_step

                if self.state.step % cfg.log.log_interval == 0:
                    t0 = self._log_train_step(loss_accum, lr, float(grad_norm), t0)

                if self.state.step % cfg.log.eval_interval == 0:
                    self._evaluate_and_checkpoint()
                    t0 = time.perf_counter()  # exclude eval time from throughput

                elif self.state.step % cfg.log.checkpoint_interval == 0:
                    self._save(f"ckpt_step{self.state.step:07d}.pt")

                self._maybe_save_milestone()
        except TrainingDiverged:
            # Do not write a final checkpoint: the weights are already poisoned, and
            # overwriting `final.pt` would destroy the last good state. `best.pt` and
            # the rolling checkpoints from before the divergence are left intact.
            self.diverged = True
            if self.dist.is_main:
                self.logger.log(self.state.step, {"train/diverged": 1})
                self.logger.close()
            raise
        else:
            # Every rank, not just the main one: the final eval runs `evaluate()`, which
            # ends in an `all_reduce_mean` over the whole group. Guarding the call with
            # `is_main` left the other ranks to exit while rank 0 waited on a collective
            # that could no longer complete, and the run hung after its last step with no
            # `final.pt` written. Only reachable when the last step misses an eval boundary
            # — `gpt2-124m` is 19073 steps at an interval of 250, so: the documented run.
            # Saving inside is already `is_main`-guarded, so this writes nothing extra.
            self._evaluate_and_checkpoint(final=True)
            if self.dist.is_main:
                self.logger.close()

        return self.state

    def _assert_finite(self, loss: float, grad_norm: float) -> None:
        """Stop the moment the loss or gradient stops being a number.

        Without this a single NaN propagates into the weights and every checkpoint
        written afterwards is poisoned, so the last recoverable state can be
        thousands of steps back by the time anyone notices. Failing on the first bad
        step keeps the blast radius to one step, and gives the ablation sweep a
        divergence it can record as a result rather than a crash.
        """
        if math.isfinite(loss) and math.isfinite(grad_norm):
            return
        raise TrainingDiverged(
            f"non-finite value at step {self.state.step:,}: "
            f"loss={loss}, grad_norm={grad_norm}. "
            f"Most often too high a learning rate (lr={self.cfg.optim.lr:g}) "
            f"or a bad batch."
        )

    def _accumulate_gradients(self) -> float:
        """Run ``grad_accum_steps`` micro-batches and return the mean loss.

        The returned value is averaged across ranks. Without that it would be rank 0's
        own micro-batches only — one Nth of the effective batch — so the logged loss
        would get noisier as the world size grew, and a multi-GPU run's curve could not
        be compared against a single-GPU one. The optimisation is unaffected either way
        (DDP averages the *gradients* regardless); this is about the number being a
        description of the batch that was actually trained on. It costs one scalar
        all-reduce per optimiser step, against a 124M-parameter gradient all-reduce.
        """
        model = self.model
        loss_accum = 0.0

        for micro_step in range(self.grad_accum_steps):
            x, y = self.train_loader.next_batch()

            # DDP all-reduces gradients on every backward. During accumulation only
            # the last micro-step needs syncing; the rest run under no_sync, which
            # removes grad_accum-1 collectives per step. At 32 accumulation steps
            # this is the difference between communication-bound and compute-bound.
            sync = micro_step == self.grad_accum_steps - 1
            ctx = model.no_sync() if (self.dist.enabled and not sync) else nullcontext()

            with ctx, autocast_context(self.device, self.dtype):
                out = model(x, targets=y)
                # Normalise so the accumulated gradient is the mean over the whole
                # effective batch, not its sum.
                loss = out.loss / self.grad_accum_steps

            loss.backward()
            loss_accum += loss.detach().float().item()

        if self.dist.enabled:
            reduced = all_reduce_mean(
                torch.tensor(loss_accum, device=self.device, dtype=torch.float32), self.dist
            )
            loss_accum = reduced.item()

        return loss_accum

    def _log_train_step(self, loss: float, lr: float, grad_norm: float, t0: float) -> float:
        cfg = self.cfg
        elapsed = time.perf_counter() - t0
        tokens = cfg.train.tokens_per_step * cfg.log.log_interval
        tokens_per_sec = tokens / elapsed

        metrics: dict[str, Any] = {
            "train/loss": loss,
            "train/lr": lr,
            "train/grad_norm": grad_norm,
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/step_time_ms": elapsed / cfg.log.log_interval * 1000,
            "progress/tokens": self.state.tokens_seen,
        }

        hardware_peak = peak_flops(self.device, self.dtype)
        if hardware_peak:
            metrics["perf/mfu"] = unwrap_model(self.model).estimate_mfu(
                tokens_per_sec, hardware_peak
            )

        if self.dist.is_main:
            mfu = f" mfu {metrics['perf/mfu'] * 100:.1f}%" if "perf/mfu" in metrics else ""
            print(
                f"step {self.state.step:>7,}/{cfg.train.max_steps:,} | "
                f"loss {loss:.4f} | lr {lr:.2e} | gnorm {grad_norm:.2f} | "
                f"{tokens_per_sec:,.0f} tok/s | "
                f"{metrics['perf/step_time_ms']:.0f} ms/step{mfu}"
            )
        self.logger.log(self.state.step, metrics)
        return time.perf_counter()

    # -------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self, steps: int | None = None) -> float:
        """Mean validation loss over ``steps`` micro-batches.

        Evaluation always starts from the beginning of the val split, so the same
        tokens are scored at every checkpoint and the curve is comparable across
        steps and across runs.
        """
        steps = steps or self.cfg.log.eval_steps
        model = self.model
        model.eval()
        self.val_loader.reset()

        total = torch.zeros((), device=self.device)
        for _ in range(steps):
            x, y = self.val_loader.next_batch()
            with autocast_context(self.device, self.dtype):
                total += model(x, targets=y).loss.detach()

        mean = total / steps
        mean = all_reduce_mean(mean, self.dist)
        model.train()
        return mean.item()

    def _evaluate_and_checkpoint(self, final: bool = False) -> None:
        # The loop always evaluates once more on exit; skip the redundant pass when
        # the last training step already landed on an eval boundary. This branch runs on
        # every rank — it contains no collective — so the write has to be guarded here,
        # the same way the writes at the bottom of the method are.
        if final and self.state.last_eval_step == self.state.step:
            if self.dist.is_main:
                self._save("final.pt", val_loss=self.state.history[-1]["val_loss"])
            return

        val_loss = self.evaluate()
        self.state.last_eval_step = self.state.step
        is_best = val_loss < self.state.best_val_loss
        if is_best:
            self.state.best_val_loss = val_loss

        if self.dist.is_main:
            print(
                f"step {self.state.step:>7,} | val loss {val_loss:.4f} | "
                f"ppl {math.exp(min(val_loss, 20)):.2f}"
                f"{'  <- best' if is_best else ''}"
            )
        self.logger.log(
            self.state.step,
            {"val/loss": val_loss, "val/perplexity": math.exp(min(val_loss, 20))},
        )
        self.state.history.append({"step": self.state.step, "val_loss": val_loss})

        if not self.dist.is_main:
            return
        if is_best:
            self._save("best.pt", val_loss=val_loss)
        if final:
            self._save("final.pt", val_loss=val_loss)
        else:
            self._save(f"ckpt_step{self.state.step:07d}.pt", val_loss=val_loss)
            prune_checkpoints(self.run_dir, self.cfg.log.keep_last_n)

    def _maybe_save_milestone(self) -> None:
        """Write a permanent checkpoint the first time the run crosses each milestone.

        Named ``milestone_*.pt`` so it does not match the ``ckpt_step*`` glob that
        pruning uses, and therefore survives for the life of the run directory.
        """
        if not self.dist.is_main or not self.cfg.log.milestone_fracs:
            return
        for frac in self.cfg.log.milestone_fracs:
            target = max(1, int(self.cfg.train.max_steps * frac))
            # Fires once, on the exact step, so a resumed run does not rewrite them.
            if self.state.step == target:
                self._save(f"milestone_{int(frac * 100):03d}pct_step{target:07d}.pt")

    def _save(self, name: str, val_loss: float | None = None) -> None:
        save_checkpoint(
            self.run_dir / name,
            self.model,
            self.optimizer,
            self.state.step,
            self.cfg,
            metrics={
                "val_loss": val_loss,
                "best_val_loss": self.state.best_val_loss,
                "tokens_seen": self.state.tokens_seen,
            },
        )
