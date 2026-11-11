# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Versioned, JSON-safe evidence produced by the offline dataset audit."""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class Finding:
    """One observation; certainty and severity are independent fields."""

    check_id: str
    classification: str
    severity: str
    location: dict[str, Any]
    expected: Any
    observed: Any
    measurements: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    limits: str = ""
    next_action: str = ""


@dataclass
class AuditReport:
    """Audit evidence with explicit completeness and command exit semantics."""

    schema_version: str = "1"
    source: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    complete: bool = True
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        check_id: str,
        classification: str,
        severity: str,
        location: dict[str, Any],
        expected: Any,
        observed: Any,
        measurements: dict[str, Any] | None = None,
        coverage: dict[str, Any] | None = None,
        limits: str = "",
        next_action: str = "",
    ) -> Finding:
        """Append a finding without silently changing audit completeness."""
        finding = Finding(
            check_id,
            classification,
            severity,
            location,
            expected,
            observed,
            measurements or {},
            coverage or {},
            limits,
            next_action,
        )
        self.findings.append(finding)
        return finding

    @property
    def exit_code(self) -> int:
        """Return 2 for incomplete audits, 1 for faults, and 0 otherwise."""
        if not self.complete:
            return 2
        return int(any(f.classification == "verified_fault" for f in self.findings))

    def to_dict(self) -> dict[str, Any]:
        """Serialize using a stable schema, including effective exit status."""
        return _json_safe({**asdict(self), "exit_code": self.exit_code})

    def to_markdown(self) -> str:
        """Render evidence without copying dataset tables or recording content."""
        report = self.to_dict()
        lines = [
            "# LeRobot dataset audit",
            "",
            f"Status: **{'complete' if self.complete else 'partial'}**; exit code: {self.exit_code}.",
            "",
            f"Input: `{self.source.get('root', self.source.get('repo_id', 'unresolved'))}`.",
            f"Dataset format: `{self.dataset.get('codebase_version', 'unknown')}`; report schema: {self.schema_version}.",
            "",
            "## Input and implementation",
            "",
            "```json",
            json.dumps(
                {
                    "source": report["source"],
                    "environment": report["environment"],
                    "dataset": report["dataset"],
                },
                indent=2,
            ),
            "```",
            "",
            "## Coverage and settings",
            "",
            "```json",
            json.dumps({"coverage": report["coverage"], "settings": report["settings"]}, indent=2),
            "```",
            "",
            "## Findings",
            "",
        ]
        for finding in report["findings"]:
            lines.extend(
                [
                    f"### {finding['check_id']} — {finding['classification']} ({finding['severity']})",
                    "",
                    f"Location: `{json.dumps(finding['location'], ensure_ascii=False)}`.",
                    "",
                    f"Expected: {finding['expected']}",
                    "",
                    f"Observed: {finding['observed']}",
                    "",
                ]
            )
            for key in ("measurements", "coverage", "limits", "next_action"):
                if finding[key]:
                    value = (
                        json.dumps(finding[key], ensure_ascii=False)
                        if isinstance(finding[key], dict)
                        else finding[key]
                    )
                    lines.extend([f"{key.replace('_', ' ').capitalize()}: {value}", ""])
        lines.extend(
            [
                "This report checks structural integrity over the stated coverage. It does not establish demonstration quality, capture synchronization, action/state alignment, or robot readiness.",
                "",
            ]
        )
        return "\n".join(lines)

    def write(self, output_dir: str | Path) -> None:
        """Write reports outside the source dataset; never modify input artifacts."""
        destination = Path(output_dir).expanduser().resolve()
        root = self.source.get("root")
        if root and destination.is_relative_to(Path(root).resolve()):
            raise ValueError("The output directory must be outside the dataset root.")
        destination.mkdir(parents=True, exist_ok=True)
        outputs = {
            "audit.json": json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            "audit.md": self.to_markdown(),
        }
        for name, content in outputs.items():
            # Atomic replacement does not follow an existing output symlink or
            # overwrite a source inode through a hard link.
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=destination, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                os.replace(temporary, destination / name)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
