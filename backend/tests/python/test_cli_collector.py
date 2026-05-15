import json
import sys
import zipfile
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.cli import main, _resolve_workload_command
import fournex as at

from analysis_bottleneck_golden_cases import INPUT_BOUND_EVENTS


def _test_out_dir(name: str) -> Path:
    path = ROOT / "traces" / "cli_test_runs" / f"{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


PTX_SIMPLE = """
.version 8.0
.target sm_80
.address_size 64

.visible .entry simple_kernel() {
    .reg .f32   %f<32>;
    .reg .b64   %rd<8>;
    ld.global.f32   %f0, [%rd0];
    fma.rn.f32      %f1, %f0, %f0, %f0;
    st.global.f32   [%rd0], %f1;
    ret;
}
"""


PTX_SPILL = """
.version 8.0
.target sm_80

.visible .entry spill_kernel() {
    .reg .f32   %f<256>;
    .reg .b64   %SP;
    .local .align 4 .b8 __local_depot0[512];
    ld.local.f32    %f0, [%SP+0];
    st.local.f32    [%SP+8], %f0;
    ret;
}
"""


CUDA_SOURCE = r"""
__global__ void bad_barrier(float* y) {
  if (threadIdx.x == 0) {
    __syncthreads();
  }
  y[threadIdx.x] = 1.0f;
}

void launch(float* y) {
  bad_barrier<<<1, 128>>>(y);
}
"""


NCU_MEMORY_BOUND = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,88.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,28.0",
    "ker,lts__t_sector_hit_rate.pct,%,38.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,55.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,45.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard,%,22.0",
])


NCU_OPTIMIZED = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,42.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,72.0",
    "ker,lts__t_sector_hit_rate.pct,%,80.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,75.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,8.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard,%,4.0",
])


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


def test_analyze_accepts_collected_zip_bundle_json(capsys) -> None:
    out_dir = _test_out_dir("analyze-zip-json")
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'python')!r})\n"
        "import fournex as at\n"
        "at.init()\n"
        "with at.step_context(step=1):\n"
        "    with at.phase('forward', step=1):\n"
        "        pass\n"
    )

    collect_code = main([
        "collect",
        "--run-id",
        "run-zip-json",
        "--out",
        str(out_dir),
        "--",
        sys.executable,
        "-c",
        code,
    ])
    capsys.readouterr()
    analyze_code = main(["analyze", str(out_dir / "run-zip-json.zip"), "--json"])
    captured = capsys.readouterr()

    assert collect_code == 0
    assert analyze_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "run_bundle"
    assert "event_count" in payload["result"]
    assert "diagnosis" in payload["result"] or "run" in payload["result"]


def test_analyze_accepts_ptx_file_json(capsys) -> None:
    out_dir = _test_out_dir("analyze-ptx")
    ptx_path = out_dir / "kernel.ptx"
    ptx_path.write_text(PTX_SPILL, encoding="utf-8")

    exit_code = main(["analyze", str(ptx_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "ptx"
    assert payload["result"]["schema"] == "ptx_analysis_v1"
    assert payload["result"]["primary_bottleneck"] == "ptx_register_spills"


def test_analyze_accepts_cuda_source_file_json(capsys) -> None:
    out_dir = _test_out_dir("analyze-cu")
    source_path = out_dir / "kernel.cu"
    source_path.write_text(CUDA_SOURCE, encoding="utf-8")

    exit_code = main(["analyze", str(source_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "cuda_source"
    assert payload["result"]["schema_version"] == "cuda_static_v1"
    assert payload["result"]["kernel_count"] == 1


def test_analyze_accepts_ncu_csv_human_report(capsys) -> None:
    out_dir = _test_out_dir("analyze-ncu")
    csv_path = out_dir / "ncu.csv"
    csv_path.write_text(NCU_MEMORY_BOUND, encoding="utf-8")

    exit_code = main(["analyze", str(csv_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Nsight Compute Analysis" in captured.out
    assert "Primary Bottleneck : memory_bandwidth_bound" in captured.out
    assert "TOP RECOMMENDATIONS" in captured.out


def test_analyze_compares_ncu_before_after_json(capsys) -> None:
    out_dir = _test_out_dir("compare-ncu")
    before = out_dir / "before.csv"
    after = out_dir / "after.csv"
    before.write_text(NCU_MEMORY_BOUND, encoding="utf-8")
    after.write_text(NCU_OPTIMIZED, encoding="utf-8")

    exit_code = main(["analyze", "--before", str(before), "--after", str(after), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "comparison"
    assert payload["result"]["schema"] == "ncu_comparison_v1"
    assert payload["result"]["verdict"]["outcome"] == "improved"


def test_analyze_compares_ptx_before_after_human_report(capsys) -> None:
    out_dir = _test_out_dir("compare-ptx")
    before = out_dir / "before.ptx"
    after = out_dir / "after.ptx"
    before.write_text(PTX_SPILL, encoding="utf-8")
    after.write_text(PTX_SIMPLE, encoding="utf-8")

    exit_code = main(["analyze", "--before", str(before), "--after", str(after)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "CUDA Before/After Comparison" in captured.out
    assert "Winner" in captured.out
    assert "Resolved: register_spills_detected" in captured.out


def test_analyze_compares_layer_specific_inputs_json(capsys) -> None:
    out_dir = _test_out_dir("compare-layered")
    before = out_dir / "before.ptx"
    after = out_dir / "after.ptx"
    before.write_text(PTX_SPILL, encoding="utf-8")
    after.write_text(PTX_SIMPLE, encoding="utf-8")

    exit_code = main(["analyze", "--before-ptx", str(before), "--after-ptx", str(after), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "comparison"
    assert payload["result"]["schema"] == "comparison_v1"


def test_analyze_rejects_unsupported_file(capsys) -> None:
    out_dir = _test_out_dir("analyze-bad-input")
    path = out_dir / "notes.txt"
    path.write_text("not cuda", encoding="utf-8")

    exit_code = main(["analyze", str(path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "unsupported input file" in captured.err


def test_ncu_command_prints_preset_command(capsys) -> None:
    exit_code = main(["ncu-command", "memory", "--output", "ncu_memory.csv", "--", "./app", "--batch", "32"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Fournex NCU command" in captured.out
    assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in captured.out
    assert "ncu --csv" in captured.out
    assert "> ncu_memory.csv" in captured.out


def test_ncu_command_lists_presets_json(capsys) -> None:
    exit_code = main(["ncu-command", "--list", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "ncu_presets"
    names = {preset["name"] for preset in payload["result"]["presets"]}
    assert {"memory", "tensor", "occupancy", "stalls", "full"} <= names


def test_analyze_ncu_malformed_csv_returns_error(capsys) -> None:
    out_dir = _test_out_dir("analyze-malformed-ncu")
    path = out_dir / "bad.csv"
    path.write_text("not,ncu\n1,2", encoding="utf-8")

    exit_code = main(["analyze", str(path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "CSV VALIDATION" in captured.out
    assert "missing a kernel name column" in captured.out


def test_analyze_accepts_root_layout_zip_bundle(capsys) -> None:
    out_dir = _test_out_dir("analyze-root-zip")
    run_dir = out_dir / "root-layout"
    summary_path = run_dir / "derived" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(at.summarize_run(INPUT_BOUND_EVENTS)), encoding="utf-8")
    zip_path = out_dir / "root-layout.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(summary_path, "derived/summary.json")

    exit_code = main(["analyze", str(zip_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Primary Bottleneck : input_bound" in captured.out


def test_analyze_rejects_malformed_zip_bundle(capsys) -> None:
    out_dir = _test_out_dir("analyze-bad-zip")
    zip_path = out_dir / "not-a-bundle.zip"
    zip_path.write_text("not a zip", encoding="utf-8")

    exit_code = main(["analyze", str(zip_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "invalid zip bundle" in captured.err


def test_analyze_rejects_zip_slip_member(capsys) -> None:
    out_dir = _test_out_dir("analyze-zip-slip")
    zip_path = out_dir / "unsafe.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.txt", "owned")

    exit_code = main(["analyze", str(zip_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "unsafe zip member path" in captured.err
    assert not (out_dir / "evil.txt").exists()
