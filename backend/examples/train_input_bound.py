"""Real input-bound (dataloader-starved) training workload.

Positive control for Fournex's bottleneck classifier on live telemetry: a
deliberately slow `Dataset` (CPU sleep per item, num_workers=0 so the stall is
in the hot path) feeds a tiny GPU step. The dataloader-fetch wait dominates each
step, so `frx analyze` should report `input_bound` with dataloader-wait evidence
and the GPU should sit mostly idle.

Run under collection:
    frx collect --name input-bound --out ~/frx_runs -- python backend/examples/train_input_bound.py
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import fournex as frx

STEPS = 40
BATCH = 8
DIM = 1024
PER_ITEM_STALL_S = 0.02  # CPU work per sample -> ~0.16s fetch wait per batch


class SlowDataset(Dataset):
    def __len__(self) -> int:
        return STEPS * BATCH

    def __getitem__(self, idx: int):
        # Simulate a slow CPU transform / decode that starves the GPU.
        time.sleep(PER_ITEM_STALL_S)
        return torch.randn(DIM), idx % DIM


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("train_input_bound requires a CUDA device")
    device = "cuda:0"
    torch.manual_seed(0)

    frx.init(job_name="input-bound-mlp")

    model = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(), nn.Linear(DIM, DIM)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # num_workers=0 keeps the stall in the main thread so the fetch wait is real
    # and visible (the realistic fix would be workers/prefetch — that's the point).
    loader = DataLoader(SlowDataset(), batch_size=BATCH, num_workers=0)
    loader = frx.instrument_dataloader(loader, loader_name="slow-cpu-loader")

    step = 0
    for inputs, targets in loader:
        if step >= STEPS:
            break
        with frx.step_context(step=step, batch={"x": inputs}, model=model) as ctx:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with frx.phase("forward", step=ctx["step"], parent_span_id=ctx["span_id"], device=device):
                loss = loss_fn(model(inputs), targets)
                torch.cuda.synchronize()
            with frx.phase("backward", step=ctx["step"], parent_span_id=ctx["span_id"], device=device):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.cuda.synchronize()
            with frx.phase("optimizer", step=ctx["step"], parent_span_id=ctx["span_id"], device=device):
                optimizer.step()
                torch.cuda.synchronize()
        step += 1

    frx.flush()


if __name__ == "__main__":
    main()
