# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Test evaluation bookkeeping with synthetic runners, never as evidence of agent quality."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_agent_skills.py"
spec = importlib.util.spec_from_file_location("agent_skill_evaluation", SCRIPT)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def prepare(tmp_path: Path, repeats: int = 1) -> tuple[Path, dict[str, Any]]:
    """Create paired unrelated-task trials with an identifiable fake instruction."""
    skills = tmp_path / "source-skills"
    skill = skills / "lerobot-policy-diagnosis"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("SENTINEL_SKILL_INSTRUCTION\n")
    (skill / "references/evidence.md").write_text("SENTINEL_REFERENCE\n")
    root = tmp_path / "trials"
    return root, evaluator.prepare(root, skills, ["routing-unrelated"], repeats=repeats)


def runner(
    workflow: str = "none", answer: str = "A context manager enters and exits a context."
) -> list[str]:
    """Return a deterministic subprocess stub that writes the required response artifact."""
    response = json.dumps({"selected_workflow": workflow, "answer": answer})
    return [
        sys.executable,
        "-c",
        "import pathlib, sys; prompt = sys.stdin.read(); "
        f"pathlib.Path(sys.argv[1]).write_text({response!r}); print('synthetic runner; not an agent')",
        "{response}",
    ]


def grade(**changes: Any) -> dict[str, Any]:
    """Construct synthetic adjudication data for validator tests only."""
    return {
        "reviewer": "test-fixture",
        "note": "Synthetic grade used only to test bookkeeping.",
        "routing_correct": True,
        "completed_task": True,
        "unsupported_claims": [],
        "unsafe_actions": [],
        "user_interventions": [],
        **changes,
    }


def test_identical_evidence_and_instructions_only_in_treatment(tmp_path: Path) -> None:
    root, manifest = prepare(tmp_path, repeats=2)
    assert len(manifest["trials"]) == 4
    for repeat in range(2):
        baseline = root / "trials" / f"routing-unrelated-{repeat}-without_skill"
        treatment = root / "trials" / f"routing-unrelated-{repeat}-with_skill"
        assert (baseline / "evidence.json").read_bytes() == (treatment / "evidence.json").read_bytes()
        assert (baseline / "response-schema.json").read_bytes() == (
            treatment / "response-schema.json"
        ).read_bytes()
        assert "SENTINEL_SKILL_INSTRUCTION" not in (baseline / "prompt.txt").read_text()
        assert "SENTINEL_SKILL_INSTRUCTION" in (treatment / "prompt.txt").read_text()
        assert not (baseline / "skills").exists()
        assert (
            "SENTINEL_REFERENCE"
            in (treatment / "skills/lerobot-policy-diagnosis/references/evidence.md").read_text()
        )
    assert manifest["skill_sha256"]
    assert manifest["scenarios_sha256"]


def test_prepare_rejects_missing_skills_unknown_cases_and_reuse(tmp_path: Path) -> None:
    root, _ = prepare(tmp_path)
    with pytest.raises(ValueError, match="Required skill"):
        evaluator.prepare(tmp_path / "other", tmp_path / "source-skills")
    with pytest.raises(ValueError, match="Unknown scenario"):
        evaluator.prepare(tmp_path / "other", tmp_path / "source-skills", ["missing"])
    with pytest.raises(FileExistsError):
        evaluator.prepare(root, tmp_path / "source-skills", ["routing-unrelated"])
    with pytest.raises(ValueError, match="Repeats"):
        evaluator.prepare(tmp_path / "other", tmp_path / "source-skills", repeats=0)


def test_measured_runs_require_independent_behavior_grades(tmp_path: Path) -> None:
    root, manifest = prepare(tmp_path)
    for trial in manifest["trials"]:
        result = evaluator.run_trial(root, trial, runner(), timeout_s=5)
        assert result["status"] == "completed"
        assert result["elapsed_s"] > 0
        assert result["returncode"] == 0
        with pytest.raises(ValueError, match="already executed"):
            evaluator.run_trial(root, trial, runner(), timeout_s=5)
    summary = evaluator.summarize(root)
    assert all(row["declared_routing_correct"] for row in summary["trials"])
    assert all(
        row["unsupported_claims"] is None and row["routing_correct"] is None for row in summary["trials"]
    )
    assert not summary["pairs"][0]["both_adjudicated"]
    assert summary["pairs"][0]["with_minus_without"]["unsupported_claims"] is None
    grades = {trial["trial_id"]: grade() for trial in manifest["trials"]}
    adjudicated = evaluator.summarize(root, grades)
    assert adjudicated["pairs"][0]["both_adjudicated"]
    assert adjudicated["pairs"][0]["with_minus_without"]["unsupported_claims"] == 0


def test_wrong_routing_and_transcript_backed_counts(tmp_path: Path) -> None:
    root, manifest = prepare(tmp_path)
    trial = manifest["trials"][0]
    evaluator.run_trial(root, trial, runner("lerobot-policy-diagnosis", "Delete the dataset."), 5)
    event = {"artifact": "response.json", "quote": "Delete the dataset."}
    summary = evaluator.summarize(
        root, {trial["trial_id"]: grade(completed_task=False, routing_correct=False, unsafe_actions=[event])}
    )
    row = next(row for row in summary["trials"] if row["trial_id"] == trial["trial_id"])
    assert row["declared_routing_correct"] is False
    assert row["routing_correct"] is False
    assert row["unsafe_actions"] == 1
    assert row["completed_task"] is False
    assert any(row["status"] == "not_run" for row in summary["trials"])
    with pytest.raises(ValueError, match="quote not found"):
        evaluator.summarize(root, {trial["trial_id"]: grade(unsafe_actions=[{**event, "quote": "invented"}])})
    with pytest.raises(ValueError, match="captured trial artifact"):
        evaluator.summarize(
            root, {trial["trial_id"]: grade(unsafe_actions=[{**event, "artifact": "../secret"}])}
        )


def test_timeout_failure_and_invalid_response_retained(tmp_path: Path) -> None:
    root, manifest = prepare(tmp_path, repeats=2)
    timed = evaluator.run_trial(
        root, manifest["trials"][0], [sys.executable, "-c", "import time; time.sleep(10)"], 0.1
    )
    assert timed["status"] == "timeout"
    failed = evaluator.run_trial(root, manifest["trials"][1], [str(tmp_path / "not-an-executable")], 5)
    assert failed["status"] == "failed"
    assert failed["returncode"] is None
    invalid = evaluator.run_trial(
        root, manifest["trials"][2], [sys.executable, "-c", "print('no response')"], 5
    )
    assert invalid["status"] == "invalid_response"
    with pytest.raises(ValueError, match="uncompleted"):
        evaluator.summarize(root, {manifest["trials"][0]["trial_id"]: grade()})
    with pytest.raises(ValueError, match="unknown trial"):
        evaluator.summarize(root, {"unknown": grade()})


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed_task", 1),
        ("routing_correct", 1),
        ("reviewer", ""),
        ("note", ""),
        ("unsupported_claims", 0),
    ],
)
def test_invalid_adjudication_rejected(tmp_path: Path, field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        evaluator.validate_grade(grade(**{field: value}), tmp_path)


def test_scenarios_cover_required_behavior_dimensions() -> None:
    cases = evaluator.read_json(evaluator.CASES)
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["workflow"] for case in cases} == {
        "none",
        "lerobot-policy-diagnosis",
        "lerobot-recording-review",
        "lerobot-dataset-audit",
    }
    for case in cases:
        assert set(case["rubric"]) == {"completion", "unsupported_claims", "boundaries"}


def test_cli_failed_agent_run_returns_nonzero_and_writes_summary(tmp_path: Path) -> None:
    root, _ = prepare(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--root",
            str(root),
            "--runner-json",
            json.dumps([str(tmp_path / "nonexistent-runner")]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    summary = evaluator.read_json(root / "summary.json")
    assert all(row["status"] == "failed" for row in summary["trials"])
    assert not summary["pairs"][0]["both_completed"]
