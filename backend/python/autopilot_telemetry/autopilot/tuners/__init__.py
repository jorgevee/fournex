from __future__ import annotations

from typing import Any

from ..actions import CandidateConfig
from . import batch_size, dataloader, mixed_precision


def generate_all_candidates(
    environment: dict[str, Any] | None = None,
    baseline_batch_size: int | None = None,
    max_total: int = 12,
    safe_only: bool = True,
) -> list[CandidateConfig]:
    """
    Staged search (section 13 of safe_autopilot_actions.md):
      Stage 1 — dataloader  (Tier 0, always)
      Stage 2 — batch size  (Tier 1, skipped when safe_only=True pending baseline)
      Stage 3 — mixed prec  (Tier 1, skipped when safe_only=True and no CUDA)

    Callers that want Tier-1 candidates should pass safe_only=False and
    supply baseline_batch_size from the baseline trial metrics.
    """
    from ..actions import TIER_SAFE

    candidates: list[CandidateConfig] = []

    dl = dataloader.generate_candidates(environment=environment, max_candidates=8)
    candidates.extend(dl)

    if not safe_only and len(candidates) < max_total:
        bs = batch_size.generate_candidates(
            baseline_batch_size=baseline_batch_size,
            environment=environment,
            max_candidates=max_total - len(candidates),
        )
        candidates.extend(bs)

    if not safe_only and len(candidates) < max_total:
        amp = mixed_precision.generate_candidates(environment=environment)
        candidates.extend(amp[:max_total - len(candidates)])

    return candidates[:max_total]
