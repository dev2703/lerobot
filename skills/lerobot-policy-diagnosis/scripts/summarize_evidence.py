# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Summarize explicitly extracted policy contracts; never load models or raw logs."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 1_048_576
MAX_TRACE_ROWS = 128
MAX_FEATURES = 64
TOKEN = re.compile(r"[A-Za-z0-9_.:/+-]{1,128}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
RULES = {
    "checkpoint_weights_sha256": ("digest", "high", "Recompute all intended/evaluated weight digests."),
    "normalization_state_sha256": (
        "digest",
        "high",
        "Compare saved normalization tensor keys/values and an offline round trip.",
    ),
    "camera_keys": ("names", "high", "Replay one observation through the saved camera renamer."),
    "image_shapes": ("shapes", "high", "Inspect per-sample shapes at the same processor boundary."),
    "state_joint_order": (
        "names",
        "high",
        "Trace one synthetic state vector through the documented mapping.",
    ),
    "action_joint_order": (
        "names",
        "high",
        "Trace one synthetic action vector through the documented mapping.",
    ),
    "state_units": ("units", "high", "Compare documented units and one offline conversion per joint."),
    "action_units": ("units", "high", "Compare documented units and one offline conversion per joint."),
    "action_convention": (
        "convention",
        "high",
        "Trace absolute/relative transforms on a fixed offline vector.",
    ),
    "gripper_convention": ("gripper", "high", "Trace gripper endpoints through the offline action mapping."),
    "action_dim": ("dimension", "high", "Inspect the unpadded physical action dimensions after processing."),
    "task_present": ("boolean", "medium", "Inspect task presence and tokenizer output on one offline batch."),
    "control_hz": (
        "rate",
        "medium",
        "Compare dispatch intervals, observation age and the execution contract.",
    ),
}
ENUMS = {
    "units": {"radians", "degrees", "meters", "normalized", "mixed"},
    "convention": {"absolute", "relative", "velocity", "mixed"},
    "gripper": {"open_positive", "open_negative", "binary_open_one", "binary_open_zero"},
}
CONFIG_INTS = {"n_obs_steps", "chunk_size", "n_action_steps", "max_state_dim", "max_action_dim", "num_steps"}


def require(condition: bool) -> None:
    """Reject malformed evidence without echoing potentially private input values."""
    if not condition:
        raise ValueError("Invalid evidence schema; see references/evidence.md")


def finite_number(value: Any, minimum: float = 0) -> bool:
    """Accept finite numeric measurements, excluding booleans and overflowing integers."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= minimum
    except OverflowError:
        return False


def positive_int(value: Any) -> bool:
    """Check bounded physical dimensions and configuration counts."""
    return type(value) is int and 0 < value <= 1_000_000


def names(value: Any) -> bool:
    """Check bounded unique feature names without accepting free-form text."""
    return (
        isinstance(value, list)
        and 0 < len(value) <= MAX_FEATURES
        and all(isinstance(v, str) and TOKEN.fullmatch(v) is not None for v in value)
        and len(set(value)) == len(value)
    )


def validate_value(kind: str, value: Any) -> None:
    """Validate a comparison using its documented value type."""
    if kind == "digest":
        valid = isinstance(value, str) and SHA256.fullmatch(value) is not None
    elif kind == "names":
        valid = names(value)
    elif kind == "shapes":
        valid = (
            isinstance(value, dict)
            and names(list(value))
            and all(
                isinstance(shape, list) and 1 <= len(shape) <= 4 and all(positive_int(v) for v in shape)
                for shape in value.values()
            )
        )
    elif kind in ENUMS:
        valid = isinstance(value, str) and value in ENUMS[kind]
    elif kind == "dimension":
        valid = positive_int(value)
    elif kind == "boolean":
        valid = type(value) is bool
    else:
        valid = finite_number(value) and value > 0
    require(bool(valid))


def sanitize(raw: Any) -> dict[str, Any]:
    """Select bounded numeric/enum evidence; exclude raw logs, paths and task text."""
    require(isinstance(raw, dict))
    require(type(raw.get("schema_version")) is int and raw["schema_version"] == 1)
    require(raw.get("policy_family") in ("act", "smolvla"))
    clean: dict[str, Any] = {"schema_version": 1, "policy_family": raw["policy_family"]}
    versions = raw.get("versions", {})
    require(isinstance(versions, dict))
    clean["versions"] = {}
    for key in ("lerobot", "torch", "source_revision"):
        if key in versions:
            value = versions[key]
            pattern = r"[a-f0-9]{7,40}" if key == "source_revision" else r"[0-9][A-Za-z0-9.+_-]{0,63}"
            require(isinstance(value, str) and re.fullmatch(pattern, value) is not None)
            clean["versions"][key] = value
    if "imports_match_checkout" in versions:
        require(type(versions["imports_match_checkout"]) is bool)
        clean["versions"]["imports_match_checkout"] = versions["imports_match_checkout"]

    config = raw.get("configuration", {})
    require(isinstance(config, dict))
    clean["configuration"] = {}
    for key in sorted(CONFIG_INTS | {"temporal_ensemble_coeff"}):
        if key in config:
            value = config[key]
            if key in CONFIG_INTS:
                require(positive_int(value))
            else:
                require(value is None or finite_number(value, minimum=-math.inf))
            clean["configuration"][key] = value

    comparisons = raw.get("comparisons", {})
    require(isinstance(comparisons, dict) and set(comparisons) <= set(RULES))
    clean["comparisons"] = {}
    for key, pair in comparisons.items():
        require(isinstance(pair, dict) and "expected" in pair and "observed" in pair)
        for side in ("expected", "observed"):
            validate_value(RULES[key][0], pair[side])
        if key == "camera_keys":  # Set membership, unlike ordered joint vectors.
            pair = {side: sorted(pair[side]) for side in ("expected", "observed")}
        clean["comparisons"][key] = {side: pair[side] for side in ("expected", "observed")}

    trace = raw.get("trace", [])
    require(isinstance(trace, list) and len(trace) <= MAX_TRACE_ROWS)
    clean["trace"] = []
    for row in trace:
        require(isinstance(row, dict))
        selected = {}
        for key in ("episode_index", "step", "queue_depth_before_select"):
            value = row.get(key)
            require(type(value) is int and 0 <= value <= 1_000_000_000)
            selected[key] = value
        for key in ("episode_start", "reset_before_select"):
            require(type(row.get(key)) is bool)
            selected[key] = row[key]
        clean["trace"].append(selected)
    return clean


def summarize(raw: Any) -> dict[str, Any]:
    """Rank discrepancies in supplied facts without claiming physical causality."""
    evidence = sanitize(raw)
    findings: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []

    def add(
        check: str,
        classification: str,
        severity: str,
        expected: Any,
        observed: Any,
        location: str,
        experiment: str,
    ) -> None:
        findings.append(
            {
                "check_id": f"policy.{check}",
                "classification": classification,
                "severity": severity,
                "location": location,
                "expected": expected,
                "observed": observed,
                "units": "Hz"
                if check == "control_hz"
                else "As supplied; inspect the physical mapping separately.",
                "tolerance": "Exact comparison of supplied values; no physical tolerance inferred.",
                "coverage": "Supplied facts only; raw artifact provenance is not validated by this helper.",
                "next_action": experiment,
            }
        )
        if classification == "warning":
            hypotheses.append(
                {
                    "check_id": f"policy.{check}",
                    "confidence": "moderate",
                    "supporting_evidence": [location],
                    "contradicting_evidence": [],
                    "missing_evidence": ["Raw artifact provenance and actual runtime path validation"],
                    "interpretation": "A supplied discrepancy may explain failure; causality is unverified.",
                    "experiment": experiment,
                    "expected_outcomes": {
                        "discrepancy_persists": "Supports a contract mismatch; task-failure causality remains open.",
                        "mapping_or_equal_values_explains_it": "Weakens this hypothesis; inspect the next boundary.",
                    },
                    "permissions": "Local offline inspection within existing scope; no hardware or training.",
                    "resource_class": "cpu_small",
                }
            )

    for key, (_, severity, experiment) in RULES.items():
        pair = evidence["comparisons"].get(key)
        location = f"failure-bundle.json#/evidence/comparisons/{key}"
        if pair is None:
            add(key, "not_checked", severity, None, None, location, experiment)
            continue
        expected, observed = pair["expected"], pair["observed"]
        same = expected == observed
        # A non-language ACT path need not consume task input.
        if key == "task_present" and evidence["policy_family"] == "act":
            add(
                key,
                "not_checked",
                "low",
                expected,
                observed,
                location,
                "Inspect whether this ACT runtime has any task-dependent preprocessing.",
            )
            continue
        add(key, "passed" if same else "warning", severity, expected, observed, location, experiment)

    for index, row in enumerate(evidence["trace"]):
        if row["episode_start"]:
            # A missing reset or nonempty pre-selection queue is suspicious, not proof of stale actions.
            mismatch = not row["reset_before_select"] or row["queue_depth_before_select"] != 0
            add(
                "episode_reset",
                "warning" if mismatch else "passed",
                "high",
                {"reset_before_select": True, "queue_depth_before_select": 0},
                {k: row[k] for k in ("reset_before_select", "queue_depth_before_select")},
                f"failure-bundle.json#/evidence/trace/{index}",
                "Replay two episodes offline through the actual reset/select path; inspect stale actions.",
            )
    if not any(row["episode_start"] for row in evidence["trace"]):
        add(
            "episode_reset",
            "not_checked",
            "high",
            None,
            None,
            "failure-bundle.json#/evidence/trace",
            "Inspect an episode-boundary trace before first action selection.",
        )

    for key in ("offline_predictions", "dispatch_and_measured_state", "physical_conditions"):
        add(
            key,
            "not_checked",
            "medium",
            None,
            None,
            "outside_helper_coverage",
            "Follow references/diagnosis.md; this helper does not execute or visually inspect these stages.",
        )
    priority = {f"policy.{key}": index for index, key in enumerate(RULES)}
    hypotheses.sort(key=lambda h: priority.get(h["check_id"], len(priority)))
    for rank, hypothesis in enumerate(hypotheses, 1):
        hypothesis["rank"] = rank
    return {
        "schema_version": 1,
        "status": "evidence_summary",
        "evidence": evidence,
        "coverage": {
            "comparison_pairs": len(evidence["comparisons"]),
            "trace_rows": len(evidence["trace"]),
            "raw_artifacts_validated": False,
            "model_executed": False,
            "hardware_accessed": False,
        },
        "findings": findings,
        "ranked_hypotheses": hypotheses,
        "limits": [
            "No closed-loop success or causal conclusion is established.",
            "No hypotheses means no supplied discrepancy, not a clean diagnosis.",
            "Contradictory evidence must be added by the inspecting agent; empty lists mean none supplied.",
            "Digests differ by bytes; compare decoded normalization values before declaring a fault.",
            "Rates are compared exactly in Hz; differing rates need contract/latency inspection.",
            "Units marked mixed/normalized and gripper labels require per-joint mapping inspection.",
        ],
    }


def read_evidence(path: Path) -> dict[str, Any]:
    """Bound input reads and reject ambiguous/nonfinite JSON."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result)
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        require(False)

    with path.open("rb") as stream:
        content = stream.read(MAX_INPUT_BYTES + 1)
    require(len(content) <= MAX_INPUT_BYTES)
    parsed = json.loads(content, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    return sanitize(parsed)


def write_report(report: dict[str, Any], destination: Path, source: Path) -> None:
    """Create a fresh bundle outside the evidence directory, without copying source files."""
    source, destination = source.resolve(), destination.resolve()
    require(not destination.is_relative_to(source.parent))
    # Prepare all bounded output before creating any files. Never overwrite an existing bundle.
    bundle = {"schema_version": 1, "evidence": report["evidence"], "coverage": report["coverage"]}
    serialized = json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report = {**report, "bundle_sha256": hashlib.sha256(serialized.encode()).hexdigest()}
    diagnosis = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    require(len(serialized.encode()) + len(diagnosis.encode()) <= 4 * MAX_INPUT_BYTES)
    lines = [
        "# Policy diagnosis evidence",
        "",
        "Status: evidence summary; physical cause remains unverified.",
        "",
    ]
    for finding in report["findings"]:
        lines.extend(
            [
                f"- `{finding['check_id']}`: {finding['classification']} ({finding['severity']}); "
                f"`{finding['location']}`. {finding['next_action']}"
            ]
        )
        if finding["classification"] != "not_checked":
            lines.append(
                f"  Expected `{json.dumps(finding['expected'], sort_keys=True)}`; "
                f"observed `{json.dumps(finding['observed'], sort_keys=True)}`. "
                f"Units: {finding['units']} Tolerance: {finding['tolerance']}"
            )
    lines.extend(["", "## Ranked hypotheses", ""])
    for hypothesis in report["ranked_hypotheses"]:
        lines.extend(
            [
                f"{hypothesis['rank']}. `{hypothesis['check_id']}` (moderate confidence). "
                f"{hypothesis['experiment']}"
            ]
        )
        lines.append(
            f"   Supporting: {', '.join(hypothesis['supporting_evidence'])}. "
            "Contradicting: none supplied. Missing: raw artifact provenance and runtime validation."
        )
        lines.append(
            "   Outcomes: persistent discrepancy supports mismatch; an intentional mapping or equal "
            "decoded values weakens it. Causality remains open. "
            f"Permission: {hypothesis['permissions']} Resources: {hypothesis['resource_class']}."
        )
    lines.extend(
        ["", "Full values, outcomes, permissions, coverage and missing evidence are in diagnosis.json.", ""]
    )
    lines.extend(report["limits"])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "failure-bundle.json").write_text(serialized, encoding="utf-8")
    (destination / "diagnosis.json").write_text(diagnosis, encoding="utf-8")
    (destination / "diagnosis.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Write a bounded private summary from explicitly selected local evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(read_evidence(args.evidence))
        write_report(report, args.output_dir, args.evidence)
    except (OSError, ValueError, TypeError, RecursionError):
        parser.exit(
            2,
            "Could not summarize evidence: invalid/unavailable input or output. See references/evidence.md.\n",
        )
    return 0  # Summary produced, not a task-success or fault-free exit code.


if __name__ == "__main__":
    raise SystemExit(main())
