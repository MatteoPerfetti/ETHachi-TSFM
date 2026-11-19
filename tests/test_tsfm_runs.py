import json

import pytest

from src.tsfm_runs import read_record, write_record


def test_run_record_accepts_verified_legacy_and_rejects_incomplete(tmp_path):
    record = {"base_checkpoint": "base", "base_revision": "abc", "reloaded_validation": {"RMSE": 1}}
    write_record(tmp_path / "run.json", record)
    assert read_record(tmp_path) == record
    record["status"] = "running"
    write_record(tmp_path / "run.json", record)
    with pytest.raises(ValueError, match="not completed"):
        read_record(tmp_path)
    record["status"] = "completed"
    record.pop("reloaded_validation")
    write_record(tmp_path / "run.json", record)
    with pytest.raises(ValueError, match="reload"):
        read_record(tmp_path)


def test_atomic_record_does_not_replace_valid_json_on_serialization_error(tmp_path):
    path = tmp_path / "run.json"
    write_record(path, {"status": "running"})
    with pytest.raises(ValueError):
        write_record(path, {"invalid": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "running"}
