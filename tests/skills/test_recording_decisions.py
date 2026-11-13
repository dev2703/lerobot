# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Check human decision validation, counts, coverage, and source preservation."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "skills/lerobot-recording-review/scripts/review_decisions.py"
spec = importlib.util.spec_from_file_location("review_decisions", SCRIPT)
decisions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decisions)


@pytest.fixture
def report() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "sampled",
        "coverage": {"sampled_frames": 3, "semantic_reviewed_frames": 0},
        "queue": [
            {"candidate_id": key, "classification": kind, "disposition": "pending"}
            for key, kind in (
                ("dark", "hypothesis"),
                ("contrast", "hypothesis"),
                ("visibility", "not_checked"),
            )
        ],
        "samples": [{"episode_index": 0, "camera": "cam", "episode_time_s": t} for t in (0.0, 0.5, 1.0)],
        "clips": [
            {
                "clip_id": "clip-0",
                "episode_index": 0,
                "camera": "cam",
                "actual_interval_s": [0.0, 0.5],
                "status": "complete",
                "clip_path": "clips/clip-0.mp4",
            }
        ],
    }


def test_counts_and_inspection_are_separate_and_idempotent(report: dict[str, Any]) -> None:
    original = copy.deepcopy(report)
    supplied = {
        "decisions": [
            {"candidate_id": "dark", "disposition": "accepted", "note": "Obstruction confirmed."},
            {"candidate_id": "contrast", "disposition": "rejected", "note": "Uniform background."},
        ],
        "reviewed_samples": [report["samples"][1]],
        "reviewed_clips": ["clip-0"],
    }
    result = decisions.apply_decisions(report, supplied)
    assert report == original
    assert result["decisions_summary"]["candidates"] == {
        "accepted": 1,
        "rejected": 1,
        "pending": 1,
        "adjudicated": 2,
        "total": 3,
        "acceptance_fraction": 0.5,
    }
    assert result["decisions_summary"]["flags"]["pending"] == 0
    assert result["coverage"]["semantic_reviewed_frames"] == 1
    assert result["coverage"]["semantic_reviewed_clips"] == 1
    assert result["coverage"]["semantic_reviewed_intervals"][0]["actual_interval_s"] == [0.0, 0.5]
    assert decisions.apply_decisions(result, supplied) == result


def test_dispositions_do_not_imply_visual_coverage(report: dict[str, Any]) -> None:
    supplied = [
        {"candidate_id": "visibility", "disposition": "accepted", "note": "Please inspect this flag."}
    ]
    result = decisions.apply_decisions(report, supplied)
    assert result["coverage"]["semantic_reviewed_frames"] == 0
    assert result["coverage"]["semantic_reviewed_clips"] == 0
    assert result["decisions_summary"]["flags"]["acceptance_fraction"] is None
    reset = decisions.apply_decisions(result, [])
    assert reset["decisions_summary"]["candidates"]["pending"] == 3


@pytest.mark.parametrize(
    "supplied",
    [
        None,
        {"decision": []},
        {"schema_version": "99"},
        {"decisions": "accepted"},
        [{"candidate_id": "unknown", "disposition": "accepted", "note": "seen"}],
        [{"candidate_id": "dark", "disposition": "yes", "note": "seen"}],
        [{"candidate_id": "dark", "disposition": "accepted", "note": " "}],
        [{"candidate_id": "dark", "disposition": "accepted", "note": "seen", "reviewer": ""}],
        [{"candidate_id": "dark", "disposition": "accepted", "note": "seen"}] * 2,
        {"reviewed_samples": [{"episode_index": 0, "camera": "cam", "episode_time_s": 0.75}]},
        {"reviewed_samples": [{"episode_index": True, "camera": "cam", "episode_time_s": 0}]},
        {"reviewed_samples": [{"episode_index": 0, "camera": "cam", "episode_time_s": float("nan")}]},
        {"reviewed_samples": [{"episode_index": 0, "camera": "cam", "episode_time_s": 0}] * 2},
        {"reviewed_clips": ["unknown"]},
        {"reviewed_clips": ["clip-0", "clip-0"]},
    ],
)
def test_invalid_decisions_are_rejected(report: dict[str, Any], supplied: Any) -> None:
    with pytest.raises(ValueError):
        decisions.apply_decisions(report, supplied)


def test_unexported_or_partial_clips_cannot_count_as_reviewed(report: dict[str, Any]) -> None:
    report["clips"][0]["status"] = "partial"
    with pytest.raises(ValueError, match="complete exported clip"):
        decisions.apply_decisions(report, {"reviewed_clips": ["clip-0"]})


def test_cli_preserves_sources_and_rejects_stale_or_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, Any]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    report["source_root"] = str(dataset)
    report_path = source / "review.json"
    decision_path = source / "decisions.json"
    report_path.write_text(json.dumps(report))
    decision_path.write_text("[]")
    originals = report_path.read_bytes(), decision_path.read_bytes()
    output = tmp_path / "adjudicated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--report",
            str(report_path),
            "--decisions",
            str(decision_path),
            "--output-dir",
            str(output),
        ],
    )
    assert decisions.main() == 0
    result = json.loads((output / "review.json").read_text())
    assert result["decisions_summary"]["candidates"]["pending"] == 3
    assert report_path.read_bytes() == originals[0] and decision_path.read_bytes() == originals[1]
    with pytest.raises(SystemExit) as exc:
        decisions.main()
    assert exc.value.code == 2
    decision_path.write_text(json.dumps({"report_sha256": "stale", "decisions": []}))
    with pytest.raises(SystemExit) as exc:
        decisions.main()
    assert exc.value.code == 2


def test_output_inside_dataset_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, Any]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    report["source_root"] = str(dataset)
    report_path = source / "review.json"
    decision_path = source / "decisions.json"
    report_path.write_text(json.dumps(report))
    decision_path.write_text("[]")
    output = dataset / "adjudicated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--report",
            str(report_path),
            "--decisions",
            str(decision_path),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        decisions.main()
    assert exc.value.code == 2 and not output.exists()
