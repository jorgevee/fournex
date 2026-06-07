"""Tests for `frx init` — guided onboarding command."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.cli import main, _detect_training_script, _already_instrumented, _patch_script


# ── _detect_training_script ───────────────────────────────────────────────────

def test_detects_train_py(tmp_path):
    (tmp_path / "train.py").write_text("# training script")
    assert _detect_training_script(tmp_path).name == "train.py"


def test_detects_train_underscore_prefix(tmp_path):
    (tmp_path / "train_resnet.py").write_text("# training script")
    result = _detect_training_script(tmp_path)
    assert result is not None
    assert result.name == "train_resnet.py"


def test_detects_main_py(tmp_path):
    (tmp_path / "main.py").write_text("# main")
    assert _detect_training_script(tmp_path).name == "main.py"


def test_returns_none_when_no_script(tmp_path):
    assert _detect_training_script(tmp_path) is None


# ── _already_instrumented ────────────────────────────────────────────────────

def test_already_instrumented_detects_import():
    assert _already_instrumented("import fournex as frx\nfrx.init()")

def test_already_instrumented_detects_frx_init():
    assert _already_instrumented("frx.init(job_name='x')")

def test_not_instrumented_plain_script():
    assert not _already_instrumented("import torch\nfor step in range(10): pass")


# ── _patch_script ─────────────────────────────────────────────────────────────

def test_patch_already_instrumented_skips(tmp_path, capsys):
    p = tmp_path / "train.py"
    p.write_text("import fournex as frx\nfrx.init()\n")
    _patch_script(p)
    out = capsys.readouterr().out
    assert "already" in out
    # File must not be modified
    assert p.read_text() == "import fournex as frx\nfrx.init()\n"


def test_patch_inserts_after_imports(tmp_path, monkeypatch, capsys):
    p = tmp_path / "train.py"
    p.write_text("import torch\nimport os\n\nfor step in range(10):\n    pass\n")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    _patch_script(p)
    content = p.read_text()
    assert "import fournex as frx" in content
    assert "frx.init" in content
    # Existing imports must still be present
    assert "import torch" in content


def test_patch_declined_leaves_file_unchanged(tmp_path, monkeypatch):
    p = tmp_path / "train.py"
    original = "import torch\nfor step in range(10): pass\n"
    p.write_text(original)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    _patch_script(p)
    assert p.read_text() == original


# ── CLI: frx init exits 0 ─────────────────────────────────────────────────────

def test_init_exits_zero(capsys):
    rc = main(["init"])
    assert rc == 0


def test_init_output_mentions_frx(capsys):
    main(["init"])
    out = capsys.readouterr().out
    assert "frx" in out.lower()


def test_init_patch_missing_file_exits_one(capsys):
    rc = main(["init", "--patch", "does_not_exist.py"])
    assert rc == 1


def test_init_patch_already_instrumented(tmp_path, capsys):
    p = tmp_path / "train.py"
    p.write_text("import fournex as frx\nfrx.init()\n")
    rc = main(["init", "--patch", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "already" in out
