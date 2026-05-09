from __future__ import annotations

import json
import pathlib
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parent / "python"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fournex.analysis import derive_step_metrics, summarize_step_scope
from fournex.cuda_static import inspect_cuda_source

app = FastAPI(title="Fournex API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FEEDBACK_PATH = pathlib.Path(__file__).parent / "data" / "feedback.jsonl"
_FEEDBACK_PATH.parent.mkdir(exist_ok=True)


class AnalyzeRequest(BaseModel):
    events: list[dict[str, Any]]
    environment: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    run_id: str
    rec_id: str
    outcome: str  # "accepted" | "rejected" | "applied" | "not_applicable"
    notes: str | None = None


class CudaSourceUpload(BaseModel):
    filename: str
    content: str


class CudaStaticInspectRequest(BaseModel):
    files: list[CudaSourceUpload]
    gpu_model: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        per_step = derive_step_metrics(request.events)
        result = summarize_step_scope(
            request.events,
            per_step=per_step,
            environment=request.environment,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/feedback")
def record_feedback(request: FeedbackRequest) -> dict[str, Any]:
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": request.run_id,
        "rec_id": request.rec_id,
        "outcome": request.outcome,
        "notes": request.notes,
    }
    with _FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"ok": True, "id": record["id"]}


@app.post("/cuda/static-inspect")
def inspect_cuda_static(request: CudaStaticInspectRequest) -> dict[str, Any]:
    if not request.files:
        raise HTTPException(status_code=422, detail="at least one CUDA source file is required")

    reports = [
        inspect_cuda_source(file.content, filename=file.filename, gpu_model=request.gpu_model)
        for file in request.files
    ]
    findings = [finding for report in reports for finding in report["findings"]]
    return {
        "schema_version": "cuda_static_upload_v1",
        "file_count": len(request.files),
        "gpu_model": request.gpu_model,
        "reports": reports,
        "summary": {
            "kernel_count": sum(report["kernel_count"] for report in reports),
            "launch_count": sum(report["launch_count"] for report in reports),
            "finding_count": len(findings),
            "high_severity_count": sum(1 for finding in findings if finding["severity"] == "high"),
        },
    }


@app.get("/feedback")
def get_feedback() -> dict[str, Any]:
    if not _FEEDBACK_PATH.exists():
        return {"feedback": []}
    records = [
        json.loads(line)
        for line in _FEEDBACK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {"feedback": records}
