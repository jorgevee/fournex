"""Real GPU-compute-bound training workload (instrumented with the Fournex SDK).

Unlike the FakeTensor demos, this does genuine CUDA work: a 3-layer MLP doing
forward/backward/optimizer on a resident batch with no dataloader in the hot
path. It exists to verify that Fournex's live telemetry path classifies a
healthy, GPU-saturated loop correctly — i.e. it should NOT raise an input/copy/
sync bottleneck, and GPU utilization sampled by `frx collect` should be high.

Run under collection:
    frx collect --name gpu-bound --out ~/frx_runs -- python backend/examples/train_gpu_bound.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

import fournex as frx

STEPS = 120
BATCH = 4096
DIM = 4096


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("train_gpu_bound requires a CUDA device")
    device = "cuda:0"
    torch.manual_seed(0)

    frx.init(job_name="gpu-bound-mlp")

    model = nn.Sequential(
        nn.Linear(DIM, DIM), nn.ReLU(),
        nn.Linear(DIM, DIM), nn.ReLU(),
        nn.Linear(DIM, DIM),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Resident batch: no host->device copy or dataloader wait in the loop.
    x = torch.randn(BATCH, DIM, device=device)
    y = torch.randint(0, DIM, (BATCH,), device=device)

    for step in range(STEPS):
        with frx.step_context(step=step, batch={"x": x}, model=model) as ctx:
            with frx.phase("forward", step=ctx["step"], parent_span_id=ctx["span_id"], device=device):
                logits = model(x)
                loss = loss_fn(logits, y)
                torch.cuda.synchronize()
            with frx.phase("backward", step=ctx["step"], parent_span_id=ctx["span_id"], device=device):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.cuda.synchronize()
            with frx.phase("optimizer", step=ctx["step"], parent_span_id=ctx["span_id"], device=device):
                optimizer.step()
                torch.cuda.synchronize()

    frx.flush()


if __name__ == "__main__":
    main()
