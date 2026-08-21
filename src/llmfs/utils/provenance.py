"""Record what produced a number.

A benchmark result without its hardware, driver and commit is not a measurement, it
is an anecdote — and the one thing guaranteed about a rented pod is that it will not
exist when someone asks "what did you run this on?". Every artifact this repo
publishes carries this block.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from typing import Any

import torch


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _public_hostname() -> str:
    """A hostname that attests without identifying.

    A rented pod's hostname is an anonymous container id, and recording it is the
    point of this block. A personal machine's default hostname is its owner's name
    with dots in it: macOS stamps `<Owner>s-MacBook-Pro.local`. Artifacts are
    published, so `.local` names are recorded by shape rather than by value.
    """
    name = platform.node()
    return "personal-workstation.local" if name.endswith(".local") else name


def measure_matmul_tflops(device: torch.device, size: int = 8192, iters: int = 20) -> float | None:
    """Achieved bf16 matmul throughput.

    Reported alongside MFU so a disappointing utilisation can be attributed: a low
    MFU against a healthy matmul rate is the model's shape or the data loader, while
    a low matmul rate is the card or the build.
    """
    if device.type != "cuda":
        return None
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        a @ a
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ a
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del a
    torch.cuda.empty_cache()
    return iters * 2 * size**3 / elapsed / 1e12


def capture(
    device: torch.device | str = "cpu",
    measure: bool = True,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> dict[str, Any]:
    """Everything needed to attribute a result to a machine and a commit.

    ``seed`` and ``deterministic`` are recorded when the caller has them: until they
    were, no benchmark artifact recorded either, and the seed survived only inside
    checkpoints and run configs — one directory-cleanup away from an unattributable
    number.
    """
    device = torch.device(device)
    info: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": _public_hostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit does not describe what ran. Better to record
        # the fact than to imply a clean provenance that is not there.
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    if seed is not None:
        info["seed"] = seed
    if deterministic is not None:
        info["deterministic"] = deterministic

    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "gpu_memory_gib": round(props.total_memory / 2**30, 1),
                "gpu_arch": f"sm_{props.major}{props.minor}",
                "cuda": torch.version.cuda,
                "torch_arch_list": torch.cuda.get_arch_list(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        driver = _nvidia_smi("driver_version")
        if driver:
            info["driver"] = driver
        if measure:
            info["measured_bf16_tflops"] = round(measure_matmul_tflops(device) or 0.0, 1)

    # Rented pods vary in core count, which shows up in data-prep timings.
    info["cpu_count"] = os.cpu_count()
    return info


def _nvidia_smi(field: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
