"""Pointwise-heavy, launch-bound workload — eager vs torch.compile.

Tests a *real* framework behavior under Fournex's normal telemetry UX: eager
PyTorch's per-op kernel-launch overhead vs torch.compile fusion. On small tensors
a long chain of elementwise ops is dominated by launch overhead (many tiny
kernels), which torch.compile fuses into a handful.

This uses ONLY the frx SDK (init + step_context + phase). Under `frx collect`
the SDK's sampled profiler captures real CUDA kernel counts automatically and
feeds launch_bound / framework_abstraction_tax — no manual torch.profiler trace
needed.

    frx collect --name pw-eager   -- python backend/examples/train_pointwise.py --mode eager
    frx analyze <run_dir> && frx explain <run_dir>

(torch.compile needs a working Triton install; without it inductor cannot
generate kernels and the compiled model will error.)
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import fournex as frx

STEPS = 80
BATCH = 128
DIM = 256
DEPTH = 40  # pointwise blocks; ~5 elementwise ops each -> ~200 tiny kernels/fwd eager


def _coeffs(depth: int):
    g = torch.Generator().manual_seed(0)
    a = [round(0.9 + 0.2 * float(torch.rand((), generator=g)), 4) for _ in range(depth)]
    b = [round(0.05 * float(torch.rand((), generator=g)), 4) for _ in range(depth)]
    return a, b


class PointwiseStack(nn.Module):
    """A long chain of small elementwise ops — launch-overhead bound in eager."""

    def __init__(self, depth: int):
        super().__init__()
        self.a, self.b = _coeffs(depth)

    def forward(self, x):
        for i in range(len(self.a)):
            x = torch.tanh(x * self.a[i] + self.b[i])
            x = x + torch.sin(x)
            x = x * torch.sigmoid(x)
        return x


def build(mode: str, device: str):
    model = PointwiseStack(DEPTH).to(device).eval()
    if mode == "eager":
        return model
    if mode == "compile":
        return torch.compile(model)
    if mode == "compile-roh":
        return torch.compile(model, mode="reduce-overhead")
    raise SystemExit(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "compile", "compile-roh"], default="eager")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("train_pointwise requires a CUDA device")
    device = "cuda:0"
    torch.manual_seed(0)

    frx.init(job_name=f"pointwise-{args.mode}")
    model = build(args.mode, device)
    x = torch.randn(BATCH, DIM, device=device)

    # Warmup — torch.compile builds its graph on the first calls; keep it out of
    # the measured steps.
    with torch.no_grad():
        for _ in range(15):
            model(x)
        torch.cuda.synchronize()

    for step in range(STEPS):
        with frx.step_context(step=step, batch={"x": x}, model=model) as ctx:
            with frx.phase("forward", step=ctx["step"], parent_span_id=ctx["span_id"], device=device):
                with torch.no_grad():
                    model(x)
                torch.cuda.synchronize()

    frx.flush()


if __name__ == "__main__":
    main()
