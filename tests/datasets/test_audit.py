# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.audit import AuditReport, audit_dataset
from lerobot.datasets.audit.resolve import InputResolutionError, resolve_input
from lerobot.scripts.lerobot_audit_dataset import main

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = pytest.mark.dataset


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    metadata = root / "meta/episodes/chunk-000/file-000.parquet"
    data = root / "data/chunk-000/file-000.parquet"
    metadata.parent.mkdir(parents=True)
    data.parent.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 3,
        "robot_type": "synthetic",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [1], "names": ["joint"]},
            "action": {"dtype": "float32", "shape": [1], "names": ["joint"]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "length": 3,
                    "dataset_from_index": 0,
                    "dataset_to_index": 3,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                }
            ]
        ),
        metadata,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "index": i,
                    "frame_index": i,
                    "timestamp": i / 30,
                    "observation.state": [0.0],
                    "action": [0.0],
                }
                for i in range(3)
            ]
        ),
        data,
    )
    return root


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_audit_and_cli_offline_and_input_unchanged(dataset_root: Path, tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden during offline audit")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    before = _hashes(dataset_root)
    output = tmp_path / "reports"
    assert main(["--root", str(dataset_root), "--output-dir", str(output)]) == 0
    result = json.loads((output / "audit.json").read_text())
    assert result["complete"] and result["exit_code"] == 0
    assert result["schema_version"] == "1"
    assert result["coverage"]["inspected_rows"] == 3
    assert result["settings"]["offline"]
    assert result["environment"]["import_matches_checkout"]
    assert "capture synchronization" in (output / "audit.md").read_text()
    assert before == _hashes(dataset_root)


def test_verified_fault_exit_code(dataset_root: Path) -> None:
    path = dataset_root / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[1]["timestamp"] = 0.0
    pq.write_table(pa.Table.from_pylist(rows), path)
    report = audit_dataset(dataset_root)
    assert report.complete and report.exit_code == 1
    assert any(
        f.check_id == "timestamps.order" and f.classification == "verified_fault" for f in report.findings
    )


@pytest.mark.parametrize("version", ["v2.1", "v3.1", None])
def test_unsupported_format_is_incomplete(dataset_root: Path, version: str | None) -> None:
    path = dataset_root / "meta/info.json"
    info = json.loads(path.read_text())
    info["codebase_version"] = version
    path.write_text(json.dumps(info))
    report = audit_dataset(dataset_root)
    assert report.exit_code == 2 and not report.complete
    assert report.findings[0].check_id == "input.format"


def test_unsupported_storage_backend(dataset_root: Path) -> None:
    path = dataset_root / "meta/info.json"
    info = json.loads(path.read_text())
    info["storage_format"] = "lance"
    path.write_text(json.dumps(info))
    report = audit_dataset(dataset_root)
    assert report.exit_code == 2 and not report.complete
    assert report.findings[0].check_id == "input.storage_format"
    assert all(f.classification != "verified_fault" for f in report.findings)


def test_missing_feature_dtype_cannot_skip_video_checks(dataset_root: Path) -> None:
    path = dataset_root / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["observation.images.camera"] = {"shape": [12, 16, 3]}
    path.write_text(json.dumps(info))
    report = audit_dataset(dataset_root)
    assert report.exit_code == 2
    assert report.findings[0].check_id == "input.schema"
    assert report.findings[0].classification == "verified_fault"


def test_execution_failure_retains_partial_report(dataset_root: Path, monkeypatch) -> None:
    import sqlite3

    from lerobot.datasets.audit import video

    def failed(root, info, episodes, report, **kwargs):
        raise sqlite3.OperationalError("No space left on temporary index disk")

    monkeypatch.setattr(video, "audit_videos", failed)
    report = audit_dataset(dataset_root)
    assert report.exit_code == 2
    assert report.coverage["inspected_rows"] == 3
    assert report.findings[-1].check_id == "audit.incomplete"
    assert "OperationalError" in report.findings[-1].observed


@pytest.mark.parametrize("contents", ["not json", "[]", '{"codebase_version":"v3.0"}'])
def test_invalid_metadata_is_not_a_clean_pass(dataset_root: Path, contents: str) -> None:
    (dataset_root / "meta/info.json").write_text(contents)
    assert audit_dataset(dataset_root).exit_code == 2


def test_partial_cache(dataset_root: Path) -> None:
    (dataset_root / "data/chunk-000/file-000.parquet").unlink()
    report = audit_dataset(dataset_root)
    assert not report.complete and report.exit_code == 2
    assert any(f.check_id == "rows.reference" and f.classification == "not_checked" for f in report.findings)


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf"), True])
def test_invalid_tolerance_is_incomplete(dataset_root: Path, tolerance: float) -> None:
    report = audit_dataset(dataset_root, timestamp_tolerance_s=tolerance)
    assert report.exit_code == 2
    json.dumps(report.to_dict(), allow_nan=False)


def test_tolerance_override_recorded(dataset_root: Path) -> None:
    report = audit_dataset(dataset_root, timestamp_tolerance_s=0.002)
    assert report.exit_code == 0
    assert report.settings["timestamp_tolerance_overridden"]
    assert report.settings["timestamp_tolerance_s"] == 0.002


def test_interruption_saves_partial_evidence(dataset_root: Path, tmp_path: Path, monkeypatch) -> None:
    from lerobot.datasets.audit import tables

    def interrupted(root, info, report, tolerance_s):
        report.add("synthetic.observed", "verified_fault", "error", {}, 1, 0)
        report.coverage["inspected_rows"] = 2
        raise KeyboardInterrupt

    monkeypatch.setattr(tables, "audit_tables", interrupted)
    output = tmp_path / "partial"
    assert main(["--root", str(dataset_root), "--output-dir", str(output)]) == 2
    report = json.loads((output / "audit.json").read_text())
    assert report["coverage"]["inspected_rows"] == 2
    assert {f["check_id"] for f in report["findings"]} == {"synthetic.observed", "audit.interrupted"}


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_report_output_links_do_not_mutate_input(dataset_root: Path, tmp_path: Path, link_kind: str) -> None:
    report = audit_dataset(dataset_root)
    output = tmp_path / "linked-output"
    output.mkdir()
    source = dataset_root / "meta/info.json"
    before = _hashes(dataset_root)
    if link_kind == "symlink":
        (output / "audit.json").symlink_to(source)
    else:
        os.link(source, output / "audit.json")
    report.write(output)
    assert before == _hashes(dataset_root)
    assert json.loads((output / "audit.json").read_text())["schema_version"] == "1"


def test_output_inside_dataset_rejected_before_scan(dataset_root: Path) -> None:
    before = _hashes(dataset_root)
    with pytest.raises(SystemExit) as error:
        main(["--root", str(dataset_root), "--output-dir", str(dataset_root / "audit")])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="outside"):
        audit_dataset(dataset_root).write(dataset_root)
    assert before == _hashes(dataset_root)


@pytest.fixture
def cache_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    return home


def test_cache_revision_and_ambiguity(cache_home: Path) -> None:
    repo = cache_home / "lerobot/hub/datasets--someone--demo"
    first, second = repo / "snapshots/aaa", repo / "snapshots/bbb"
    first.mkdir(parents=True)
    second.mkdir()
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text("bbb")
    report = audit_dataset(repo_id="someone/demo")
    assert report.exit_code == 2
    assert report.findings[0].measurements["candidates"] == sorted([str(first), str(second)])
    assert resolve_input(None, "someone/demo", "main")[0] == second
    assert resolve_input(None, "someone/demo", "aaa")[0] == first


def test_legacy_and_global_hub_cache(cache_home: Path) -> None:
    legacy = cache_home / "lerobot/someone/demo"
    legacy.mkdir(parents=True)
    assert resolve_input(None, "someone/demo", None)[0] == legacy
    global_snapshot = cache_home / "hub/datasets--someone--demo/snapshots/aaa"
    global_snapshot.mkdir(parents=True)
    with pytest.raises(InputResolutionError, match="Several"):
        resolve_input(None, "someone/demo", None)
    assert resolve_input(None, "someone/demo", "aaa")[0] == global_snapshot


@pytest.mark.parametrize("repo_id", ["../demo", "someone/../demo", "/tmp", "demo"])
def test_invalid_repo_id_rejected(cache_home: Path, repo_id: str) -> None:
    with pytest.raises(InputResolutionError):
        resolve_input(None, repo_id, None)


def test_missing_input_and_exclusive_sources(dataset_root: Path, cache_home: Path) -> None:
    assert audit_dataset().exit_code == 2
    assert audit_dataset(dataset_root, repo_id="someone/demo").exit_code == 2
    assert audit_dataset(dataset_root, revision="main").exit_code == 2
    report = audit_dataset(repo_id="someone/missing")
    assert report.exit_code == 2
    assert "No network lookup" in report.findings[0].observed


def test_report_warning_does_not_fail() -> None:
    report = AuditReport()
    report.add("test.warning", "warning", "error", {}, "Expected", "Suspicious")
    assert report.exit_code == 0


def test_current_writer_output_and_public_dataset_api(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    pytest.importorskip("av")
    from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset

    root = tmp_path / "writer-dataset"
    dataset = LeRobotDataset.create(
        repo_id="synthetic/audit",
        root=root,
        fps=30,
        use_videos=False,
        video_backend="pyav",
        features={
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
        },
    )
    try:
        for episode in range(2):
            for frame in range(3):
                dataset.add_frame(
                    {
                        "observation.state": np.array([episode], dtype=np.float32),
                        "action": np.array([frame], dtype=np.float32),
                        "task": "synthetic audit regression",
                    }
                )
            dataset.save_episode()
    finally:
        dataset.finalize()
    report = audit_dataset(root)
    assert report.dataset["codebase_version"] == CODEBASE_VERSION
    assert report.complete and report.exit_code == 0
    assert report.coverage["inspected_episodes"] == 2
    assert report.coverage["inspected_rows"] == 6
