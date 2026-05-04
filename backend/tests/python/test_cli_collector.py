import json
import sys
import zipfile
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.cli import main, _resolve_workload_command


def _test_out_dir(name: str) -> Path:
    path = ROOT / "traces" / "cli_test_runs" / f"{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_collect_creates_run_folder_logs_manifest_and_zip() -> None:
    out_dir = _test_out_dir("success")
    code = "import os; print('run', os.environ['FRX_RUN_ID'])"

    exit_code = main([
        "collect",
        "--run-id",
        "run-test",
        "--name",
        "unit-test",
        "--out",
        str(out_dir),
        "--",
        sys.executable,
        "-c",
        code,
    ])

    assert exit_code == 0
    run_dir = out_dir / "run-test"
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "run_config.yaml").exists()
    assert (run_dir / "gpu_metrics.csv").exists()
    assert (run_dir / "optional_logs.txt").exists()
    assert (run_dir / "manifest.json").exists()
    assert (out_dir / "run-test.zip").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run-test"
    assert metadata["job_name"] == "unit-test"
    assert metadata["exit_code"] == 0
    assert "run run-test" in (run_dir / "optional_logs.txt").read_text(encoding="utf-8")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "metadata.json" in manifest["included_files"]
    assert "optional_logs.txt" in manifest["included_files"]

    with zipfile.ZipFile(out_dir / "run-test.zip") as archive:
        names = set(archive.namelist())
    assert "run-test/metadata.json" in names
    assert "run-test/manifest.json" in names


def test_collect_resolves_python_to_current_interpreter() -> None:
    assert _resolve_workload_command(["python", "train.py"]) == [sys.executable, "train.py"]
    assert _resolve_workload_command(["python.exe", "train.py"]) == [sys.executable, "train.py"]
    assert _resolve_workload_command(["python3", "train.py"]) == [sys.executable, "train.py"]


def test_collect_preserves_child_exit_code_and_artifacts() -> None:
    out_dir = _test_out_dir("failure")

    exit_code = main([
        "collect",
        "--run-id",
        "run-fail",
        "--out",
        str(out_dir),
        "--no-zip",
        "--",
        sys.executable,
        "-c",
        "import sys; print('failing'); sys.exit(7)",
    ])

    assert exit_code == 7
    run_dir = out_dir / "run-fail"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 7
    assert not (out_dir / "run-fail.zip").exists()


def test_collect_auto_persists_sdk_artifacts() -> None:
    out_dir = _test_out_dir("sdk")
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'python')!r})\n"
        "import fournex as at\n"
        "at.init()\n"
        "with at.step_context(step=1):\n"
        "    with at.phase('forward', step=1):\n"
        "        pass\n"
    )

    exit_code = main([
        "collect",
        "--run-id",
        "run-sdk",
        "--out",
        str(out_dir),
        "--no-zip",
        "--",
        sys.executable,
        "-c",
        code,
    ])

    assert exit_code == 0
    run_dir = out_dir / "run-sdk"
    assert (run_dir / "raw" / "trace.jsonl").exists()
    assert (run_dir / "derived" / "summary.json").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifacts"]["raw_trace"] == "raw/trace.jsonl"
    assert metadata["artifacts"]["derived_summary"] == "derived/summary.json"


def test_collect_imports_workload_bundle_artifacts() -> None:
    out_dir = _test_out_dir("workload-bundle")
    code = (
        "from pathlib import Path\n"
        "bundle = Path('frx-job-run')\n"
        "bundle.mkdir(exist_ok=True)\n"
        "(bundle / 'profiler_trace.json').write_text('{\"traceEvents\": []}', encoding='utf-8')\n"
        "(bundle / 'metadata.json').write_text('{\"workload\": \"smoke\"}', encoding='utf-8')\n"
        "(bundle / 'gpu_metrics.csv').write_text('timestamp,utilization.gpu\\nnow,10\\n', encoding='utf-8')\n"
    )

    exit_code = main([
        "collect",
        "--run-id",
        "run-workload-bundle",
        "--out",
        str(out_dir),
        "--no-zip",
        "--",
        sys.executable,
        "-c",
        code,
    ])

    assert exit_code == 0
    run_dir = out_dir / "run-workload-bundle"
    assert (run_dir / "profiler" / "profiler_trace.json").exists()
    assert (run_dir / "raw" / "workload_metadata.json").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifacts"]["profiler_trace"] == "profiler/profiler_trace.json"
    assert any("Imported workload artifacts" in warning for warning in metadata["collection_warnings"])

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["diagnostic_ready"] is True
    assert manifest["limited"] is False
