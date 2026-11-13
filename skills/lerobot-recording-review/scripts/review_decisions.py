#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Validate supplied human decisions and derive recording-review counts offline."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _sample_key(value: Any) -> tuple[int, str, float]:
    sample = _object(value, "Reviewed sample")
    episode, camera, time = (sample.get(k) for k in ("episode_index", "camera", "episode_time_s"))
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError("Reviewed sample requires a nonnegative episode_index")
    if not isinstance(camera, str) or not camera:
        raise ValueError("Reviewed sample requires a camera")
    if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0:
        raise ValueError("Reviewed sample requires a finite nonnegative episode_time_s")
    return episode, camera, float(time)


def apply_decisions(report: dict[str, Any], supplied: Any) -> dict[str, Any]:
    """Apply an authoritative decision snapshot without inferring visual inspection.

    The caller supplies actual human dispositions/inspection attestations. This
    validates their references, not the truth of the attestation or task success.
    Missing candidates reset to pending so repeated runs cannot inflate counts.
    """
    _object(report, "Report")
    if report.get("schema_version") != "1.0":
        raise ValueError("Unsupported review schema_version")
    if isinstance(supplied, list):
        supplied = {"decisions": supplied}
    supplied = _object(supplied, "Decisions")
    allowed = {"schema_version", "report_sha256", "decisions", "reviewed_samples", "reviewed_clips"}
    if supplied.keys() - allowed:
        raise ValueError("Unknown decisions fields")
    if supplied.get("schema_version", "1.0") != "1.0":
        raise ValueError("Unsupported decisions schema_version")
    result = copy.deepcopy(report)
    queue: dict[str, dict[str, Any]] = {}
    for item in _list(result.get("queue"), "Report queue"):
        item = _object(item, "Queue candidate")
        identifier = item.get("candidate_id")
        if not isinstance(identifier, str) or not identifier or identifier in queue:
            raise ValueError("Report has missing or duplicate candidate IDs")
        queue[identifier] = item
        item["disposition"] = "pending"
        item.pop("human_decision", None)
    seen: set[str] = set()
    for decision in _list(supplied.get("decisions", []), "Decisions"):
        decision = _object(decision, "Decision")
        if decision.keys() - {"candidate_id", "disposition", "note", "reviewer"}:
            raise ValueError("Unknown decision fields")
        identifier = decision.get("candidate_id")
        if not isinstance(identifier, str) or identifier not in queue:
            raise ValueError("Decision references an unknown candidate_id")
        if identifier in seen:
            raise ValueError("Duplicate candidate decision")
        seen.add(identifier)
        disposition, note = decision.get("disposition"), decision.get("note")
        if disposition not in ("pending", "accepted", "rejected"):
            raise ValueError("Disposition must be pending, accepted, or rejected")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("Each decision requires a nonempty human note")
        if "reviewer" in decision and (
            not isinstance(decision["reviewer"], str) or not decision["reviewer"].strip()
        ):
            raise ValueError("Reviewer must be a nonempty string when supplied")
        queue[identifier]["disposition"] = disposition
        queue[identifier]["human_decision"] = copy.deepcopy(decision)

    available_samples = {_sample_key(s) for s in _list(result.get("samples"), "Report samples")}
    reviewed_samples: set[tuple[int, str, float]] = set()
    for sample in _list(supplied.get("reviewed_samples", []), "Reviewed samples"):
        sample = _object(sample, "Reviewed sample")
        if sample.keys() != {"episode_index", "camera", "episode_time_s"}:
            raise ValueError("Reviewed samples require exactly episode_index, camera, episode_time_s")
        key = _sample_key(sample)
        if key not in available_samples:
            raise ValueError("Reviewed sample was not inspected by the recording helper")
        if key in reviewed_samples:
            raise ValueError("Duplicate reviewed sample")
        reviewed_samples.add(key)

    clips: dict[str, dict[str, Any]] = {}
    for clip in _list(result.get("clips", []), "Report clips"):
        clip = _object(clip, "Report clip")
        identifier = clip.get("clip_id")
        if not isinstance(identifier, str) or not identifier or identifier in clips:
            raise ValueError("Report has missing or duplicate clip IDs")
        clips[identifier] = clip
    reviewed_clips: list[str] = []
    for identifier in _list(supplied.get("reviewed_clips", []), "Reviewed clips"):
        if not isinstance(identifier, str) or identifier not in clips:
            raise ValueError("Reviewed clip references an unknown clip_id")
        if identifier in reviewed_clips:
            raise ValueError("Duplicate reviewed clip")
        clip = clips[identifier]
        if not clip.get("clip_path") or clip.get("status") != "complete":
            raise ValueError("Reviewed clips must have a complete exported clip")
        reviewed_clips.append(identifier)

    def counts(items: list[dict[str, Any]]) -> dict[str, Any]:
        values = {
            state: sum(i["disposition"] == state for i in items)
            for state in ("accepted", "rejected", "pending")
        }
        adjudicated = values["accepted"] + values["rejected"]
        return {
            **values,
            "total": len(items),
            "adjudicated": adjudicated,
            "acceptance_fraction": values["accepted"] / adjudicated if adjudicated else None,
        }

    result["decisions_summary"] = {
        "candidates": counts(list(queue.values())),
        "flags": counts([i for i in queue.values() if i.get("classification") == "hypothesis"]),
        "limits": "Human acceptance measures flag usefulness, not task success or model accuracy.",
    }
    coverage = _object(result.get("coverage"), "Report coverage")
    coverage["semantic_reviewed_frames"] = len(reviewed_samples)
    coverage["semantic_reviewed_clips"] = len(reviewed_clips)
    for interval in coverage.get("inspected_clip_intervals", []):
        interval["semantic_review"] = "reviewed" if interval.get("clip_id") in reviewed_clips else "pending"
    coverage["semantic_reviewed_intervals"] = [
        {k: clips[i][k] for k in ("clip_id", "episode_index", "camera", "actual_interval_s")}
        for i in reviewed_clips
    ]
    result["human_review"] = {
        "reviewed_samples": [
            {"episode_index": e, "camera": c, "episode_time_s": t} for e, c, t in sorted(reviewed_samples)
        ],
        "reviewed_clips": reviewed_clips,
        "limits": "Coverage records supplied human inspection attestations; dispositions alone add no coverage. "
        "Frame coverage counts stills only; clip coverage is separate and cannot establish unseen task phases.",
    }
    return result


def main() -> int:
    """Write a new adjudicated report; preserve original reports and recordings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report_bytes, decisions_bytes = args.report.read_bytes(), args.decisions.read_bytes()
        report, decisions = json.loads(report_bytes), json.loads(decisions_bytes)
        report_hash = hashlib.sha256(report_bytes).hexdigest()
        if (
            isinstance(decisions, dict)
            and "report_sha256" in decisions
            and decisions["report_sha256"] != report_hash
        ):
            raise ValueError("Decisions report_sha256 does not match the input report")
        updated = apply_decisions(report, decisions)
        updated["decision_sources"] = {
            "report_sha256": report_hash,
            "decisions_sha256": hashlib.sha256(decisions_bytes).hexdigest(),
        }
        # The derived report does not copy private stills/clips. Preserve the
        # origin so their relative references remain usable across adjudications.
        updated["artifact_root"] = report.get("artifact_root", str(args.report.resolve().parent))
        output = args.output_dir.expanduser().resolve()
        repository = Path(__file__).resolve().parents[3]
        source_root = report.get("source_root")
        if not isinstance(source_root, str) or not source_root:
            raise ValueError("Report requires source_root for output isolation")
        protected = (
            repository,
            Path(source_root).expanduser().resolve(),
            args.report.resolve().parent,
            args.decisions.resolve().parent,
        )
        if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
            raise ValueError(
                "Output must be a fresh directory outside source/report/decisions directories and repository"
            )
        # Validate and serialize before creating any output.
        payload = json.dumps(updated, indent=2, allow_nan=False) + "\n"
        output.mkdir(parents=True, exist_ok=False)
        (output / "review.json").write_text(payload)
        summary = updated["decisions_summary"]
        (output / "review.md").write_text(
            "# Recording review decisions\n\n"
            f"Numeric sampling status: {updated['status']}.\n\n"
            "```json\n"
            + json.dumps({"decisions": summary, "coverage": updated["coverage"]}, indent=2)
            + "\n```\n\n"
            + updated["human_review"]["limits"]
            + "\n\n"
            "Candidate evidence and human notes are retained in review.json. "
            "Resolve private frame/clip paths against artifact_root in that report.\n"
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Invalid review decisions: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
