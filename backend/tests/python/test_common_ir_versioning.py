from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.common_ir import (
    CURRENT_SCHEMA_VERSION,
    AnnotationRecord,
    EventRecord,
    MetricRecord,
    RunRecord,
    SchemaVersionError,
    _MIGRATIONS,
    migrate_record_dict,
    register_migration,
    validate_run_dict,
)
from fournex.common_ir_validators import validate_run_payload
from fournex.storage import load_run_record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_event_dict(**overrides):
    base = {
        "event_id": "e-1",
        "run_id": "r-1",
        "event_family": "cpu",
        "event_type": "step_start",
        "ts_start_ns": 0,
        "ts_end_ns": 100,
        "duration_ns": 100,
        "source": "sdk",
        "schema_version": CURRENT_SCHEMA_VERSION,
    }
    base.update(overrides)
    return base


def _minimal_run_dict(**overrides):
    base = {
        "run_id": "r-1",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "job": {
            "job_id": "j-1",
            "workload_class": "training",
            "status": "complete",
        },
        "workload": {"model_family": "transformer"},
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clear_migrations():
    """Restore _MIGRATIONS to empty after each test that registers throwaway migrations."""
    originals = {k: list(v) for k, v in _MIGRATIONS.items()}
    yield
    for k in _MIGRATIONS:
        _MIGRATIONS[k][:] = originals[k]


# ---------------------------------------------------------------------------
# WS3 tests
# ---------------------------------------------------------------------------

def test_missing_schema_version_treated_as_current():
    d = _minimal_event_dict()
    del d["schema_version"]
    rec = EventRecord.from_dict(d)
    assert rec.schema_version == CURRENT_SCHEMA_VERSION


def test_current_version_passes_through():
    d = _minimal_event_dict()
    rec = EventRecord.from_dict(d)
    assert rec.schema_version == CURRENT_SCHEMA_VERSION


def test_future_version_raises_naming_both_versions():
    d = _minimal_event_dict(schema_version="2.0.0")
    with pytest.raises(SchemaVersionError) as exc_info:
        EventRecord.from_dict(d)
    msg = str(exc_info.value)
    assert "2.0.0" in msg
    assert CURRENT_SCHEMA_VERSION in msg
    assert "upgrade fournex" in msg


def test_older_version_with_registered_migration_applies():
    def upgrade_run(data):
        data = dict(data)
        if "old_field" in data:
            data["new_field"] = data.pop("old_field")
        data["schema_version"] = CURRENT_SCHEMA_VERSION
        return data

    register_migration("run", "0.9.0", CURRENT_SCHEMA_VERSION, upgrade_run)

    d = _minimal_run_dict(schema_version="0.9.0", old_field="value")
    rec = RunRecord.from_dict(d)
    assert rec.schema_version == CURRENT_SCHEMA_VERSION


def test_older_version_with_empty_registry_raises_no_migration_path():
    d = _minimal_event_dict(schema_version="0.5.0")
    with pytest.raises(SchemaVersionError) as exc_info:
        EventRecord.from_dict(d)
    msg = str(exc_info.value)
    assert "0.5.0" in msg
    assert "No migration path" in msg


def test_malformed_version_string_raises():
    d = _minimal_event_dict(schema_version="not-a-version")
    with pytest.raises(SchemaVersionError) as exc_info:
        EventRecord.from_dict(d)
    assert "not-a-version" in str(exc_info.value)


def test_event_record_gated():
    d = _minimal_event_dict(schema_version="9.9.9")
    with pytest.raises(SchemaVersionError):
        EventRecord.from_dict(d)


def test_run_record_gated():
    d = _minimal_run_dict(schema_version="9.9.9")
    with pytest.raises(SchemaVersionError):
        RunRecord.from_dict(d)


def test_validate_run_dict_propagates_schema_error():
    d = _minimal_run_dict(schema_version="9.9.9")
    with pytest.raises(SchemaVersionError):
        validate_run_dict(d)


def test_validate_run_payload_propagates_schema_error():
    d = _minimal_run_dict(schema_version="9.9.9")
    with pytest.raises(SchemaVersionError):
        validate_run_payload(d)


def test_load_run_record_future_version_raises():
    import tempfile, os
    d = _minimal_run_dict(schema_version="9.9.9")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(d, f)
        name = f.name
    try:
        with pytest.raises(SchemaVersionError):
            load_run_record(name)
    finally:
        os.unlink(name)


def test_load_run_record_current_version_succeeds():
    import tempfile, os
    d = _minimal_run_dict()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(d, f)
        name = f.name
    try:
        rec = load_run_record(name)
        assert rec.run_id == "r-1"
        assert rec.schema_version == CURRENT_SCHEMA_VERSION
    finally:
        os.unlink(name)


def test_migrate_record_dict_future_raises():
    with pytest.raises(SchemaVersionError) as exc_info:
        migrate_record_dict("run", {"schema_version": "99.0.0"})
    assert "99.0.0" in str(exc_info.value)
    assert CURRENT_SCHEMA_VERSION in str(exc_info.value)


def test_register_migration_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown migration kind"):
        register_migration("unknown_kind", "0.1.0", "1.0.0", lambda d: d)
