# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Offline, read-only integrity auditing of LeRobot v3.0 datasets."""

import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import lerobot

from .models import AuditReport, Finding
from .resolve import InputResolutionError, resolve_input

# dataset_metadata.CODEBASE_VERSION and LeRobotDataset's default tolerance.
# Keep this lightweight entry point usable when dataset extras are missing.
SUPPORTED_FORMAT = "v3.0"
DEFAULT_TOLERANCE_S = 1e-4


def _environment() -> dict[str, Any]:
    package_path = Path(lerobot.__file__).resolve()
    checkout = package_path.parents[2]
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "lerobot_version": lerobot.__version__,
        "imported_package": str(package_path),
        "source_checkout": str(checkout),
    }
    inspected_checkout = Path.cwd()
    if not (inspected_checkout / "src/lerobot/__init__.py").is_file():
        inspected_checkout = checkout
    result["inspected_checkout"] = str(inspected_checkout)
    result["import_matches_checkout"] = (
        package_path == (inspected_checkout / "src/lerobot/__init__.py").resolve()
    )
    result["dependencies"] = {}
    for package in ("pyarrow", "av"):
        try:
            result["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["dependencies"][package] = None
    try:
        distribution = importlib.metadata.distribution("lerobot")
        result["installed_version"] = distribution.version
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            result["installed_source"] = json.loads(direct_url).get("url")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        result["installed_version"] = None
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        result["repository_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result["repository_revision"] = None
    return result


def audit_dataset(
    root: str | Path | None = None,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    video_scan: str = "full",
    sample_seed: int = 0,
    timestamp_tolerance_s: float | None = None,
) -> AuditReport:
    """Inspect existing artifacts, returning partial evidence on unsupported inputs.

    No dataset objects or Hub APIs are constructed. This function never downloads,
    repairs, trains, or touches hardware. Call ``report.write()`` to save outputs.
    """
    tolerance = DEFAULT_TOLERANCE_S if timestamp_tolerance_s is None else timestamp_tolerance_s
    report = AuditReport(
        source={"root": str(Path(root).expanduser().resolve())}
        if root is not None
        else {"repo_id": repo_id, "revision": revision},
        environment=_environment(),
        settings={
            "video_scan": video_scan,
            "sample_seed": sample_seed if video_scan in ("sample", "spot") else None,
            "timestamp_tolerance_s": tolerance,
            "timestamp_tolerance_overridden": timestamp_tolerance_s is not None,
            "offline": True,
        },
    )
    try:
        if video_scan not in ("full", "sample", "spot"):
            raise InputResolutionError("video_scan must be full, sample or spot.")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise InputResolutionError("timestamp_tolerance_s must be finite and nonnegative.")
        path, report.source = resolve_input(root, repo_id, revision)
        metadata = path / "meta/info.json"
        try:
            info = json.loads(metadata.read_text())
        except (OSError, ValueError) as exc:
            report.complete = False
            classification = "verified_fault" if isinstance(exc, ValueError) else "not_checked"
            report.add(
                "input.metadata",
                classification,
                "error",
                {"file": str(metadata)},
                "Readable local meta/info.json",
                str(exc),
                limits="A missing cached artifact does not establish remote corruption.",
                next_action="Supply a complete existing local v3.0 dataset.",
            )
            return report
        if not isinstance(info, dict):
            raise InputResolutionError("meta/info.json must contain a JSON object.")
        report.dataset = {
            key: info.get(key)
            for key in (
                "codebase_version",
                "storage_format",
                "fps",
                "total_episodes",
                "total_frames",
                "robot_type",
            )
        }
        report.coverage.update(
            {"declared_episodes": info.get("total_episodes"), "declared_rows": info.get("total_frames")}
        )
        if info.get("codebase_version") != SUPPORTED_FORMAT:
            report.complete = False
            report.add(
                "input.format",
                "not_checked",
                "error",
                {"file": "meta/info.json"},
                SUPPORTED_FORMAT,
                info.get("codebase_version"),
                next_action="Use a v3.0 dataset; migration is outside this audit.",
            )
            return report
        if (info.get("storage_format") or "lerobot") != "lerobot":
            report.complete = False
            report.add(
                "input.storage_format",
                "not_checked",
                "error",
                {"file": "meta/info.json"},
                "The default lerobot Parquet/video storage backend",
                info["storage_format"],
                next_action="Use an audit supporting this backend; conversion is outside this audit.",
            )
            return report
        fps = info.get("fps")
        invalid = []
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            invalid.append("fps must be finite and positive")
        for key in ("total_episodes", "total_frames"):
            if type(info.get(key)) is not int or info[key] < 0:
                invalid.append(f"{key} must be a nonnegative integer")
        features = info.get("features")
        if not isinstance(features, dict) or not all(isinstance(v, dict) for v in features.values()):
            invalid.append("features must be a mapping of feature definitions")
        elif not all(isinstance(v.get("dtype"), str) and v["dtype"] for v in features.values()):
            invalid.append("every feature must declare a nonempty string dtype")
        if not isinstance(info.get("data_path"), str):
            invalid.append("data_path must be a path template")
        if invalid:
            report.complete = False
            report.add(
                "input.schema",
                "verified_fault",
                "error",
                {"file": "meta/info.json"},
                "Valid v3.0 metadata needed for the supported checks",
                invalid,
                next_action="Inspect metadata against the v3.0 writer before attempting any repair.",
            )
            return report
        from .tables import audit_tables

        episodes = (
            audit_tables(path, info, report, tolerance, scan_rows=False)
            if video_scan == "spot"
            else audit_tables(path, info, report, tolerance)
        )
        from .video import audit_videos

        audit_videos(
            path,
            info,
            episodes,
            report,
            video_scan=video_scan,
            sample_seed=sample_seed,
            tolerance_s=tolerance,
        )
        report.add(
            "timestamps.capture_synchronization",
            "not_checked",
            "info",
            {},
            "Independent sensor capture timestamps to establish capture synchronization",
            "Dataset timestamps alone do not establish physical synchronization.",
            next_action="Provide independent capture timing evidence if synchronization is in question.",
        )
    except InputResolutionError as exc:
        report.complete = False
        report.add(
            "input.resolve",
            "not_checked",
            "error",
            {},
            "One supported existing local input",
            str(exc),
            measurements={"candidates": exc.candidates},
            next_action="Select an explicit local root or cached revision.",
        )
    except KeyboardInterrupt:
        report.complete = False
        report.add(
            "audit.interrupted",
            "not_checked",
            "warning",
            {},
            "Completed requested scan",
            "Interrupted by user",
            coverage=dict(report.coverage),
            next_action="Rerun to complete the scan; existing evidence is partial.",
        )
    except Exception as exc:
        # Preserve partial evidence even for decoder/Arrow/SQLite execution
        # errors with library-specific exception hierarchies.
        report.complete = False
        report.add(
            "audit.incomplete",
            "not_checked",
            "error",
            {},
            "All supported checks executable",
            f"{type(exc).__name__}: {exc}",
            coverage=dict(report.coverage),
            next_action="Resolve the reported local artifact or dependency problem and rerun.",
        )
    return report


__all__ = ["AuditReport", "Finding", "audit_dataset"]
