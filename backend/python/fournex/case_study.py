"""Case-study harness: turn a bad/good CUDA kernel pair into a validated,
artifact-emitting optimization proof.

A case study is a directory with a ``case_study.yaml`` manifest plus a baseline
(`bad`) and optimized (`good`) source file. ``run_case_study`` drives the existing
analysis stack — static source analysis, the multi-layer comparison engine, and
the explain/brief renderers — then checks the result against the manifest's
expectations and emits reproducible artifacts.

The default path uses pure static source analysis, so a case study is fully
reproducible from the repo on any machine (no GPU or CUDA toolkit required).
When Nsight Compute CSVs are supplied (``before_ncu`` / ``after_ncu`` in the
manifest), hardware-counter evidence is layered on top automatically.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .comparison import compare_implementations
from .cuda_static import inspect_cuda_source
from .explain import (
    build_explain_result,
    render_evidence_json,
    render_llm_prompt_txt,
    render_summary_txt,
)
from .ncu_analysis import analyze_ncu_csv_text

logger = logging.getLogger(__name__)

MANIFEST_NAME = "case_study.yaml"
SCHEMA = "case_study_v1"


# ── Schema ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CaseExpectation:
    detected_before: list[str] = field(default_factory=list)
    resolved_after: list[str] = field(default_factory=list)
    no_new_findings: bool = True


@dataclass(frozen=True)
class CaseStudy:
    name: str
    title: str
    category: str
    description: str
    case_dir: Path
    bad_source: str
    good_source: str
    expected: CaseExpectation
    gpu_model: str | None = None
    before_ncu: str | None = None
    after_ncu: str | None = None


def load_case_study(case_dir: str | Path) -> CaseStudy:
    """Load a case study from a directory containing ``case_study.yaml``."""
    case_dir = Path(case_dir)
    manifest = case_dir / MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {case_dir}")

    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    for key in ("name", "bad_source", "good_source"):
        if not data.get(key):
            raise ValueError(f"{manifest}: missing required field '{key}'")

    exp = data.get("expected") or {}
    expected = CaseExpectation(
        detected_before=list(exp.get("detected_before", [])),
        resolved_after=list(exp.get("resolved_after", [])),
        no_new_findings=bool(exp.get("no_new_findings", True)),
    )
    return CaseStudy(
        name=data["name"],
        title=data.get("title", data["name"]),
        category=data.get("category", "general"),
        description=(data.get("description") or "").strip(),
        case_dir=case_dir,
        bad_source=data["bad_source"],
        good_source=data["good_source"],
        expected=expected,
        gpu_model=data.get("gpu_model"),
        before_ncu=data.get("before_ncu"),
        after_ncu=data.get("after_ncu"),
    )


def discover_case_studies(root: str | Path) -> list[CaseStudy]:
    """Find every case study under *root* (any dir with a manifest), name-sorted."""
    root = Path(root)
    cases = [
        load_case_study(m.parent)
        for m in sorted(root.rglob(MANIFEST_NAME))
    ]
    return sorted(cases, key=lambda c: c.name)


# ── Run ─────────────────────────────────────────────────────────────────────

def _read(case_dir: Path, name: str | None) -> str | None:
    if not name:
        return None
    return (case_dir / name).read_text(encoding="utf-8")


def run_case_study(case: CaseStudy, *, environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyze the pair, diff it, validate against expectations, and return a result dict."""
    bad_src = _read(case.case_dir, case.bad_source) or ""
    good_src = _read(case.case_dir, case.good_source) or ""
    before_ncu_csv = _read(case.case_dir, case.before_ncu)
    after_ncu_csv = _read(case.case_dir, case.after_ncu)

    a_input: dict[str, Any] = {
        "label": "baseline", "cuda_source": bad_src,
        "cuda_filename": case.bad_source, "gpu_model": case.gpu_model,
    }
    b_input: dict[str, Any] = {
        "label": "optimized", "cuda_source": good_src,
        "cuda_filename": case.good_source, "gpu_model": case.gpu_model,
    }
    if before_ncu_csv:
        a_input["ncu_csv"] = before_ncu_csv
    if after_ncu_csv:
        b_input["ncu_csv"] = after_ncu_csv

    comparison = compare_implementations(a_input, b_input)

    before_static = inspect_cuda_source(bad_src, filename=case.bad_source, gpu_model=case.gpu_model)
    after_static = inspect_cuda_source(good_src, filename=case.good_source, gpu_model=case.gpu_model)
    before_ncu_result = (
        analyze_ncu_csv_text(before_ncu_csv, environment=environment) if before_ncu_csv else None
    )
    explain = build_explain_result(
        static_result=before_static, ncu_result=before_ncu_result, environment=environment,
    )

    before_findings = [
        {"code": f.get("code"), "severity": f.get("severity"), "message": f.get("message")}
        for f in before_static.get("findings", []) if f.get("code")
    ]
    before_codes = sorted({f["code"] for f in before_findings})
    validation = validate_case_study(case, before_codes, comparison)

    has_ncu = before_ncu_csv is not None and after_ncu_csv is not None
    logger.debug(
        "run_case_study %s: before=%s resolved=%s passed=%s",
        case.name, before_codes,
        comparison["static_diff"].get("findings_diff", {}).get("resolved_in_b"),
        validation["passed"],
    )

    return {
        "schema": SCHEMA,
        "name": case.name,
        "title": case.title,
        "category": case.category,
        "description": case.description,
        "evidence_layer": "static+ncu" if has_ncu else "static",
        "sources": {"baseline": case.bad_source, "optimized": case.good_source},
        "baseline_source": bad_src,
        "before_codes": before_codes,
        "before_findings": before_findings,
        "after_codes": sorted({f.get("code") for f in after_static.get("findings", []) if f.get("code")}),
        "comparison": comparison,
        "explain": explain,
        "validation": validation,
    }


# ── Validation ──────────────────────────────────────────────────────────────

def validate_case_study(
    case: CaseStudy, before_codes: list[str], comparison: dict[str, Any],
) -> dict[str, Any]:
    """Check the run against the manifest: detected-before, resolved-after, no-regression."""
    findings_diff = comparison.get("static_diff", {}).get("findings_diff", {})
    resolved = set(findings_diff.get("resolved_in_b", []))
    new = set(findings_diff.get("new_in_b", []))
    before = set(before_codes)

    checks: list[dict[str, Any]] = []

    missing_detect = [c for c in case.expected.detected_before if c not in before]
    checks.append({
        "check": "expected_bottlenecks_detected_before",
        "status": "pass" if not missing_detect else "fail",
        "expected": case.expected.detected_before,
        "missing": missing_detect,
    })

    missing_resolved = [c for c in case.expected.resolved_after if c not in resolved]
    checks.append({
        "check": "expected_bottlenecks_resolved_after",
        "status": "pass" if not missing_resolved else "fail",
        "expected": case.expected.resolved_after,
        "missing": missing_resolved,
    })

    if case.expected.no_new_findings:
        checks.append({
            "check": "no_new_findings_introduced",
            "status": "pass" if not new else "fail",
            "new_findings": sorted(new),
        })

    passed = all(c["status"] == "pass" for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "resolved": sorted(resolved),
        "new": sorted(new),
    }


# ── Renderers ───────────────────────────────────────────────────────────────

def _diag_lines(result: dict[str, Any]) -> list[str]:
    findings = result.get("before_findings") or []
    if not findings:
        return ["- (no findings)"]
    lines: list[str] = []
    for f in findings:
        sev = f.get("severity", "?")
        lines.append(f"- [{sev}] {f['code']}")
        if f.get("message"):
            lines.append(f"    {f['message']}")
    return lines


def _fix_lines(result: dict[str, Any]) -> list[str]:
    """Prefer catalog recommendations; fall back to the messages of resolved findings."""
    recs = result["explain"].get("top_recommendations") or []
    if recs:
        out: list[str] = []
        for r in recs[:3]:
            out.append(f"- {r.get('title', r.get('id', 'recommendation'))}")
        return out

    resolved = set(result["validation"]["resolved"])
    msgs = [
        f"- {f['message']}"
        for f in (result.get("before_findings") or [])
        if f.get("code") in resolved and f.get("message")
    ]
    return msgs or ["- (demonstrated in the optimized kernel)"]


def render_case_study_txt(result: dict[str, Any]) -> str:
    """Human-readable proof transcript."""
    v = result["validation"]
    lines: list[str] = []
    lines.append(f"CASE STUDY: {result['title']}")
    lines.append("=" * (len("CASE STUDY: ") + len(result["title"])))
    if result["description"]:
        lines += ["", result["description"]]
    lines += ["", f"Evidence layer : {result['evidence_layer']}"]

    lines += ["", "1. Baseline kernel:", f"   {result['sources']['baseline']}"]
    lines += ["", "2. Fournex diagnosis (baseline):"]
    lines += [f"   {ln}" for ln in _diag_lines(result)]

    lines += ["", "3. Recommended fix:"]
    lines += [f"   {ln}" for ln in _fix_lines(result)]

    lines += ["", "4. Optimized kernel:", f"   {result['sources']['optimized']}"]

    lines += ["", "5. Before/after:"]
    if v["resolved"]:
        lines.append(f"   resolved   : {', '.join(v['resolved'])}")
    if v["new"]:
        lines.append(f"   new        : {', '.join(v['new'])}")
    if not v["resolved"] and not v["new"]:
        lines.append("   (no finding-level change)")

    lines += ["", "6. Verdict:"]
    for c in v["checks"]:
        mark = "PASS" if c["status"] == "pass" else "FAIL"
        lines.append(f"   [{mark}] {c['check']}")
    lines.append("")
    lines.append(
        "   Fournex prediction VALIDATED."
        if v["passed"] else
        "   Fournex prediction NOT validated (see failed checks above)."
    )
    return "\n".join(lines) + "\n"


def render_case_study_readme(result: dict[str, Any]) -> str:
    """GitHub-ready markdown writeup."""
    v = result["validation"]
    status = "✅ validated" if v["passed"] else "❌ not validated"
    lines: list[str] = []
    lines.append(f"# Case Study: {result['title']}")
    lines.append("")
    lines.append(f"**Status:** {status} &nbsp;·&nbsp; **Category:** {result['category']} "
                 f"&nbsp;·&nbsp; **Evidence:** {result['evidence_layer']}")
    if result["description"]:
        lines += ["", result["description"]]

    lines += ["", "## Diagnosis (baseline)", ""]
    for c in result["before_codes"] or ["(none)"]:
        lines.append(f"- `{c}`")

    lines += ["", "## Before → After", "", "| | Findings |", "|---|---|"]
    lines.append(f"| Baseline (`{result['sources']['baseline']}`) | "
                 f"{', '.join(f'`{c}`' for c in result['before_codes']) or '—'} |")
    lines.append(f"| Optimized (`{result['sources']['optimized']}`) | "
                 f"{', '.join(f'`{c}`' for c in result['after_codes']) or '—'} |")
    if v["resolved"]:
        lines += ["", f"**Resolved:** {', '.join(f'`{c}`' for c in v['resolved'])}"]
    if v["new"]:
        lines += ["", f"**New:** {', '.join(f'`{c}`' for c in v['new'])}"]

    lines += ["", "## Validation", ""]
    for c in v["checks"]:
        mark = "✅" if c["status"] == "pass" else "❌"
        lines.append(f"- {mark} {c['check']}")

    lines += ["", "## Reproduce", "", "```bash",
              f"frx case-study run {result['name']} --emit-readme", "```", ""]
    return "\n".join(lines)


# ── Artifacts ───────────────────────────────────────────────────────────────

def emit_case_study_artifacts(
    result: dict[str, Any], out_dir: str | Path, *, emit_readme: bool = False,
) -> dict[str, str]:
    """Write the case-study artifact bundle; return {artifact: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    explain = result["explain"]

    written: dict[str, str] = {}

    def _w(name: str, text: str) -> None:
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written[name] = str(path)

    _w("case_study.txt", render_case_study_txt(result))
    _w("diagnosis.txt", render_summary_txt(explain, src_filename=result["sources"]["baseline"]))
    _w("llm_brief.txt", render_llm_prompt_txt(
        explain,
        kernel_source=result.get("baseline_source"),
        src_filename=result["sources"]["baseline"],
    ))
    _w("evidence.json", render_evidence_json(explain))
    _w("compare.json", json.dumps(result["comparison"], indent=2, default=str))
    _w("validation.json", json.dumps(result["validation"], indent=2, default=str))
    if emit_readme:
        _w("README.md", render_case_study_readme(result))
    return written
