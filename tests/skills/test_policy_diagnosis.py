# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Test evidence comparisons and private export boundaries without policy extras."""

import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "skills/lerobot-policy-diagnosis/scripts/summarize_evidence.py"
spec = importlib.util.spec_from_file_location("policy_evidence", SCRIPT)
assert spec is not None and spec.loader is not None
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def evidence(family: str = "act", **comparisons: Any) -> dict[str, Any]:
    return {"schema_version": 1, "policy_family": family, "comparisons": comparisons}


def finding(report: dict[str, Any], check: str) -> dict[str, Any]:
    return next(f for f in report["findings"] if f["check_id"] == f"policy.{check}")


@pytest.mark.parametrize(
    "check,expected,observed",
    [
        ("camera_keys", ["observation.images.top"], ["observation.images.wrist"]),
        ("normalization_state_sha256", "a" * 64, "b" * 64),
        ("checkpoint_weights_sha256", "c" * 64, "d" * 64),
        ("action_joint_order", ["shoulder", "elbow"], ["elbow", "shoulder"]),
        ("state_joint_order", ["shoulder", "elbow"], ["elbow", "shoulder"]),
        ("action_convention", "absolute", "relative"),
        ("state_units", "degrees", "radians"),
        ("action_units", "degrees", "radians"),
        ("gripper_convention", "open_positive", "open_negative"),
        ("action_dim", 6, 7),
        ("image_shapes", {"camera": [3, 480, 640]}, {"camera": [480, 640, 3]}),
        ("control_hz", 30, 15),
    ],
)
def test_supported_discrepancies_are_ranked_but_not_causal(check: str, expected: Any, observed: Any) -> None:
    report = helper.summarize(evidence(**{check: {"expected": expected, "observed": observed}}))
    assert finding(report, check)["classification"] == "warning"
    (hypothesis,) = report["ranked_hypotheses"]
    assert hypothesis["check_id"] == f"policy.{check}"
    assert hypothesis["supporting_evidence"] == [f"failure-bundle.json#/evidence/comparisons/{check}"]
    assert hypothesis["missing_evidence"] and hypothesis["contradicting_evidence"] == []
    assert hypothesis["confidence"] == "moderate"
    assert hypothesis["experiment"] and hypothesis["expected_outcomes"]
    assert hypothesis["resource_class"] == "cpu_small"


def test_identity_precedes_observation_and_timing() -> None:
    report = helper.summarize(
        evidence(
            control_hz={"expected": 30, "observed": 10},
            camera_keys={"expected": ["top"], "observed": ["wrist"]},
            checkpoint_weights_sha256={"expected": "a" * 64, "observed": "b" * 64},
        )
    )
    assert [h["check_id"] for h in report["ranked_hypotheses"]] == [
        "policy.checkpoint_weights_sha256",
        "policy.camera_keys",
        "policy.control_hz",
    ]
    assert [h["rank"] for h in report["ranked_hypotheses"]] == [1, 2, 3]


@pytest.mark.parametrize("family", ["act", "smolvla"])
def test_matching_contracts_retain_physical_ambiguity(family: str) -> None:
    report = helper.summarize(
        evidence(
            family,
            camera_keys={"expected": ["top", "wrist"], "observed": ["wrist", "top"]},
            action_joint_order={"expected": ["shoulder", "elbow"], "observed": ["shoulder", "elbow"]},
            action_convention={"expected": "absolute", "observed": "absolute"},
        )
    )
    assert report["ranked_hypotheses"] == []
    assert finding(report, "camera_keys")["classification"] == "passed"
    assert finding(report, "physical_conditions")["classification"] == "not_checked"
    assert finding(report, "offline_predictions")["classification"] == "not_checked"
    assert report["status"] == "evidence_summary"
    assert report["coverage"]["model_executed"] is False


def test_insufficient_evidence_is_never_clean_or_verified_fault() -> None:
    report = helper.summarize(evidence())
    assert {f["classification"] for f in report["findings"]} == {"not_checked"}
    assert report["coverage"]["comparison_pairs"] == 0
    assert report["ranked_hypotheses"] == []


@pytest.mark.parametrize("family,classification", [("act", "not_checked"), ("smolvla", "warning")])
def test_missing_language_is_policy_specific(family: str, classification: str) -> None:
    report = helper.summarize(evidence(family, task_present={"expected": True, "observed": False}))
    assert finding(report, "task_present")["classification"] == classification


@pytest.mark.parametrize(
    "start,reset,depth,classification",
    [
        (True, False, 4, "warning"),
        (True, True, 4, "warning"),
        (True, False, 0, "warning"),
        (True, True, 0, "passed"),
        (False, False, 4, "not_checked"),
    ],
)
def test_queue_check_respects_episode_boundary(
    start: bool, reset: bool, depth: int, classification: str
) -> None:
    raw = evidence()
    raw["trace"] = [
        {
            "episode_index": 1,
            "step": 0,
            "episode_start": start,
            "reset_before_select": reset,
            "queue_depth_before_select": depth,
        }
    ]
    report = helper.summarize(raw)
    assert finding(report, "episode_reset")["classification"] == classification
    assert report["coverage"]["trace_rows"] == 1


def test_bundle_projection_no_network_and_input_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    raw = evidence(action_dim={"expected": 6, "observed": 6, "raw_log": "PRIVATE_LOG"})
    raw.update({"token": "SECRET_TOKEN", "task": "PRIVATE_TASK", "recording": "/private/video.mp4"})
    raw["versions"] = {"lerobot": "0.5.0", "source_revision": "e6ffde603", "credential": "SECRET_TOKEN"}
    raw["configuration"] = {"chunk_size": 100, "n_action_steps": 50, "hub_token": "SECRET_TOKEN"}
    source = tmp_path / "evidence/input.json"
    source.parent.mkdir()
    source.write_text(json.dumps(raw))
    before = source.read_bytes()
    sanitized = helper.read_evidence(source)
    report = helper.summarize(sanitized)
    output = tmp_path / "reports/diagnosis"
    helper.write_report(report, output, source)
    assert source.read_bytes() == before
    assert sorted(p.name for p in output.iterdir()) == [
        "diagnosis.json",
        "diagnosis.md",
        "failure-bundle.json",
    ]
    bundle = (output / "failure-bundle.json").read_bytes()
    exported = " ".join(p.read_text() for p in output.iterdir())
    assert all(
        secret not in exported for secret in ("SECRET_TOKEN", "PRIVATE_TASK", "PRIVATE_LOG", "/private/video")
    )
    saved = json.loads((output / "diagnosis.json").read_text())
    assert saved["bundle_sha256"] == hashlib.sha256(bundle).hexdigest()
    assert json.loads(bundle)["evidence"] == sanitized
    assert saved["coverage"]["hardware_accessed"] is False


@pytest.mark.parametrize("target", ["inside", "existing", "symlink"])
def test_output_boundaries(tmp_path: Path, target: str) -> None:
    source = tmp_path / "input/evidence.json"
    source.parent.mkdir()
    source.write_text(json.dumps(evidence()))
    if target == "inside":
        destination = source.parent / "new"
    elif target == "existing":
        destination = tmp_path / "existing"
        destination.mkdir()
        (destination / "diagnosis.json").write_text("preserve")
    else:
        destination = tmp_path / "alias"
        destination.symlink_to(source.parent, target_is_directory=True)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, FileExistsError)):
        helper.write_report(helper.summarize(evidence()), destination, source)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"schema_version": True, "policy_family": "act"},
        {"schema_version": 1, "policy_family": "diffusion"},
        evidence(action_dim={"expected": 6}),
        evidence(action_dim={"expected": True, "observed": 6}),
        evidence(action_dim={"expected": 0, "observed": 6}),
        evidence(unknown={"expected": 1, "observed": 1}),
        evidence(control_hz={"expected": float("nan"), "observed": 30}),
        evidence(control_hz={"expected": float("inf"), "observed": 30}),
        evidence(control_hz={"expected": 10**400, "observed": 30}),
        evidence(camera_keys={"expected": ["top", "top"], "observed": ["top"]}),
        evidence(image_shapes={"expected": {"top": [3, -1]}, "observed": {"top": [3, 480, 640]}}),
        evidence(normalization_state_sha256={"expected": "not-a-digest", "observed": "a" * 64}),
        {**evidence(), "trace": [{}]},
        {**evidence(), "trace": [{}] * 129},
        {**evidence(), "configuration": {"chunk_size": -1}},
        {**evidence(), "versions": {"lerobot": "/private/path"}},
    ],
)
def test_malformed_evidence_is_refused(raw: Any) -> None:
    with pytest.raises(ValueError):
        helper.summarize(raw)


@pytest.mark.parametrize(
    "contents",
    [
        '{"schema_version": 1, "schema_version": 2, "policy_family": "act"}',
        " " * (helper.MAX_INPUT_BYTES + 1),
    ],
)
def test_bounded_unambiguous_json(tmp_path: Path, contents: str) -> None:
    source = tmp_path / "input.json"
    source.write_text(contents)
    with pytest.raises(ValueError):
        helper.read_evidence(source)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_constants_refused_even_in_discarded_fields(tmp_path: Path, constant: str) -> None:
    source = tmp_path / "input.json"
    source.write_text('{"schema_version": 1, "policy_family": "act", "unused": ' + constant + "}")
    with pytest.raises(ValueError):
        helper.read_evidence(source)


def test_trace_bound_is_explicit_and_maximum_bundle_is_bounded(tmp_path: Path) -> None:
    raw = evidence()
    raw["trace"] = [
        {
            "episode_index": i,
            "step": 0,
            "episode_start": True,
            "reset_before_select": False,
            "queue_depth_before_select": 4,
        }
        for i in range(128)
    ]
    report = helper.summarize(raw)
    source = tmp_path / "input/evidence.json"
    source.parent.mkdir()
    source.write_text(json.dumps(raw))
    output = tmp_path / "report"
    helper.write_report(report, output, source)
    assert report["coverage"]["trace_rows"] == 128
    assert len(report["ranked_hypotheses"]) == 128
    assert sum(p.stat().st_size for p in output.glob("*.json")) < 4 * helper.MAX_INPUT_BYTES


def test_findings_resolve_to_bundle_evidence(tmp_path: Path) -> None:
    raw = evidence(action_dim={"expected": 6, "observed": 7})
    report = helper.summarize(raw)
    source = tmp_path / "input/evidence.json"
    source.parent.mkdir()
    source.write_text(json.dumps(raw))
    output = tmp_path / "report"
    helper.write_report(report, output, source)
    bundle = json.loads((output / "failure-bundle.json").read_text())
    location = finding(report, "action_dim")["location"]
    filename, pointer = location.split("#")
    assert filename == "failure-bundle.json"
    selected = bundle
    for part in pointer.strip("/").split("/"):
        selected = selected[part]
    assert selected == raw["comparisons"]["action_dim"]


def test_invalid_cli_evidence_never_echoes_private_content_or_creates_output(tmp_path: Path) -> None:
    source = tmp_path / "input/evidence.json"
    source.parent.mkdir()
    source.write_text(json.dumps(evidence(action_dim={"expected": "SECRET_TOKEN", "observed": 6})))
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--evidence", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "SECRET_TOKEN" not in result.stderr and "Traceback" not in result.stderr
    assert not output.exists()


def test_cli_produces_private_bundle_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "input/evidence.json"
    source.parent.mkdir()
    source.write_text(json.dumps(evidence("smolvla", task_present={"expected": True, "observed": False})))
    output = tmp_path / "output"
    command = [sys.executable, str(SCRIPT), "--evidence", str(source), "--output-dir", str(output)]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    original = (output / "diagnosis.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 2
    assert (output / "diagnosis.json").read_bytes() == original
