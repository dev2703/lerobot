# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read-only checks of the v3.0 episode and tabular timing contracts."""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _pyarrow_available, require_package

from .models import AuditReport

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow.parquet as pq

_EPISODE_FIELDS = (
    "episode_index",
    "length",
    "dataset_from_index",
    "dataset_to_index",
    "data/chunk_index",
    "data/file_index",
)
_ROW_FIELDS = ("index", "episode_index", "frame_index", "timestamp")


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _data_path(root: Path, template: str, episode: dict[str, Any]) -> Path:
    relative = Path(
        template.format(chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])
    )
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Data paths must be relative to the dataset root.")
    return root / relative


@dataclass
class _Issues:
    """Bound repeated row failures to a few examples for each check and episode."""

    report: AuditReport
    tolerance_s: float
    items: dict[tuple[str, str, int | None], dict[str, Any]] = field(default_factory=dict)

    def add(
        self,
        check: str,
        path: str,
        episode: int | None,
        row: int,
        expected: Any,
        observed: Any,
    ) -> None:
        key = check, path, episode
        if key not in self.items:
            item: dict[str, Any] = {"count": 0, "examples": []}
            item["finding"] = self.report.add(
                check,
                "verified_fault",
                "error",
                {"file": path, "episode_index": episode},
                expected,
                item["examples"],
                measurements={"affected_rows": 0, "timestamp_tolerance_s": self.tolerance_s},
                coverage={"example_limit": 5},
                next_action="Inspect the listed rows and their episode metadata before training.",
            )
            self.items[key] = item
        item = self.items[key]
        item["count"] += 1
        item["finding"].measurements["affected_rows"] = item["count"]
        if len(item["examples"]) < 5:
            item["examples"].append({"row": row, "expected": expected, "observed": observed})


def _load_episodes(root: Path, report: AuditReport) -> tuple[list[dict[str, Any]], int]:
    episodes: list[dict[str, Any]] = []
    observed_count = 0
    # Match io_utils.load_nested_dataset: deeper/nonnested artifacts are not
    # episode rows that the reader would load.
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        report.complete = False
        report.add(
            "episodes.metadata",
            "not_checked",
            "error",
            {"file": "meta/episodes/**/*.parquet"},
            "Locally available v3.0 episode metadata",
            "No episode metadata parquet files found",
            next_action="Provide the missing local metadata; a cache omission does not prove remote corruption.",
        )
    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            parquet = pq.ParquetFile(path)
            missing = set(_EPISODE_FIELDS).difference(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"Missing required episode columns: {sorted(missing)}")
            file_row = 0
            for batch in parquet.iter_batches(batch_size=1024):
                for record in batch.to_pylist():
                    metadata_ordinal = observed_count
                    observed_count += 1
                    if _is_index(record["episode_index"]) and record["episode_index"] != metadata_ordinal:
                        report.add(
                            "episodes.order",
                            "verified_fault",
                            "error",
                            {"file": relative, "metadata_row": file_row},
                            {"episode_index": metadata_ordinal},
                            {"episode_index": record["episode_index"]},
                            limits="The reader accesses episode metadata by physical row position.",
                            next_action="Inspect episode ordering across the sorted metadata shards.",
                        )
                    invalid = [key for key in _EPISODE_FIELDS if not _is_index(record[key])]
                    file_row += 1
                    if invalid:
                        report.add(
                            "episodes.metadata",
                            "verified_fault",
                            "error",
                            {"file": relative, "metadata_row": file_row - 1},
                            "Nonnegative integer episode identifiers, lengths, ranges, and file indices",
                            {key: repr(record[key]) for key in invalid},
                            next_action="Inspect the invalid episode metadata fields.",
                        )
                        continue
                    episodes.append(record)
        except (OSError, ValueError, TypeError, NotImplementedError) as exc:
            report.complete = False
            report.add(
                "episodes.metadata",
                "verified_fault",
                "error",
                {"file": relative},
                "Readable v3.0 episode metadata with required columns",
                str(exc),
                next_action="Inspect this local metadata artifact; preserve the original before any repair.",
            )
    report.coverage["episode_metadata_files"] = len(files)
    return episodes, observed_count


def _check_episode_contract(
    episodes: list[dict[str, Any]], observed_count: int, info: dict[str, Any], report: AuditReport
) -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    next_start = 0
    for ordinal, episode in enumerate(sorted(episodes, key=lambda item: item["episode_index"])):
        ep_id = episode["episode_index"]
        start, end, length = episode["dataset_from_index"], episode["dataset_to_index"], episode["length"]
        if ep_id in by_id or ep_id != ordinal:
            report.add(
                "episodes.identifiers",
                "verified_fault",
                "error",
                {"episode_index": ep_id},
                {"episode_index": ordinal, "unique": True},
                {"episode_index": ep_id, "duplicate": ep_id in by_id},
                next_action="Inspect duplicate or missing episode identifiers.",
            )
        if end <= start or end - start != length or start != next_start:
            report.add(
                "episodes.ranges",
                "verified_fault",
                "error",
                {"episode_index": ep_id},
                {"dataset_from_index": next_start, "range_length": length, "nonempty": True},
                {"dataset_from_index": start, "dataset_to_index": end, "range_length": end - start},
                measurements={"declared_length_rows": length},
                next_action="Inspect the episode's half-open dataset range for gaps or overlaps.",
            )
        next_start = end
        by_id.setdefault(ep_id, episode)
    observed_lengths = sum(episode["length"] for episode in episodes)
    if observed_count < info["total_episodes"]:
        report.complete = False
        report.add(
            "episodes.available",
            "not_checked",
            "error",
            {"file": "meta/episodes"},
            info["total_episodes"],
            observed_count,
            next_action="Provide all declared episode metadata before claiming complete coverage.",
            limits="A partial cache cannot establish whether the remote dataset is corrupt.",
        )
    if observed_count != info["total_episodes"] or observed_lengths != info["total_frames"]:
        report.add(
            "episodes.totals",
            "verified_fault",
            "error",
            {"file": "meta/info.json"},
            {"episodes": info["total_episodes"], "frames": info["total_frames"]},
            {"metadata_episodes": observed_count, "summed_episode_lengths": observed_lengths},
            next_action="Compare declared totals with the locally available episode metadata.",
        )
    return by_id


def audit_tables(
    root: Path, info: dict[str, Any], report: AuditReport, tolerance_s: float
) -> list[dict[str, Any]]:
    """Check v3.0 metadata and rows without constructing or caching a Dataset.

    Row memory is bounded by a parquet batch. Episode counters and at most five
    examples per failed check/episode are retained, rather than entire tables.
    """
    try:
        require_package("pyarrow", "dataset")
    except ImportError as exc:
        report.complete = False
        report.add(
            "tables.dependencies",
            "not_checked",
            "error",
            {},
            "pyarrow available to read parquet artifacts",
            str(exc),
            next_action="Install the dataset extra in the selected environment, then rerun the audit.",
        )
        return []
    findings_before = len(report.findings)
    episodes, metadata_count = _load_episodes(root, report)
    by_id = _check_episode_contract(episodes, metadata_count, info, report)
    template = info["data_path"]
    paths_by_id: dict[int, Path] = {}
    for ep_id, episode in by_id.items():
        try:
            path = _data_path(root, template, episode)
            paths_by_id[ep_id] = path
            relative_path = path.relative_to(root)
            if (
                len(relative_path.parts) != 3
                or relative_path.parts[0] != "data"
                or relative_path.suffix != ".parquet"
            ):
                report.complete = False
                report.add(
                    "rows.layout",
                    "not_checked",
                    "error",
                    {"episode_index": ep_id, "file": relative_path.as_posix()},
                    "A data shard visible to the v3.0 reader's data/*/*.parquet discovery",
                    "The referenced data path is outside the reader's supported layout",
                    next_action="Verify the dataset layout against the current reader; do not infer compatibility from a custom template.",
                )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            report.complete = False
            report.add(
                "rows.reference",
                "verified_fault",
                "error",
                {"episode_index": ep_id, "field": "data_path"},
                "A relative data_path template using chunk_index and file_index",
                str(exc),
                next_action="Inspect the data path template and file indices.",
            )
    # The actual reader loads every parquet in this layout, including files
    # whose names do not match info.data_path (for example a stray backup).
    discovered = set((root / "data").glob("*/*.parquet"))
    data_paths = sorted(set(paths_by_id.values()) | discovered)
    report.coverage["data_files"] = len(data_paths)
    report.coverage["inspected_rows"] = 0
    report.coverage["inspected_episodes"] = 0
    counts: dict[int, int] = defaultdict(int)
    previous_ts: dict[int, float] = {}
    unavailable_episodes: set[int] = set()
    issues = _Issues(report, tolerance_s)
    complete_files = 0
    next_dataset_index = 0
    preceding_files_readable = True
    for path in data_paths:
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            report.complete = False
            preceding_files_readable = False
            unavailable_episodes.update(ep_id for ep_id, expected in paths_by_id.items() if expected == path)
            report.add(
                "rows.reference",
                "not_checked",
                "error",
                {"file": relative},
                "A locally available referenced data file",
                "Referenced parquet is missing",
                next_action="Provide the missing local artifact; remote corruption has not been established.",
            )
            continue
        physical_row = 0
        expected_file_start = min(
            (
                by_id[ep_id]["dataset_from_index"]
                for ep_id, expected in paths_by_id.items()
                if expected == path
            ),
            default=None,
        )
        if (
            preceding_files_readable
            and expected_file_start is not None
            and expected_file_start != next_dataset_index
        ):
            report.add(
                "rows.file_order",
                "verified_fault",
                "error",
                {"file": relative},
                {"first_global_index": next_dataset_index},
                {"metadata_first_global_index": expected_file_start},
                next_action="Inspect shard ordering: the reader concatenates data files in sorted path order.",
            )
        try:
            parquet = pq.ParquetFile(path)
            missing = set(_ROW_FIELDS).difference(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"Missing required data columns: {sorted(missing)}")
            for batch in parquet.iter_batches(batch_size=8192, columns=list(_ROW_FIELDS)):
                for row in batch.to_pylist():
                    report.coverage["inspected_rows"] += 1
                    if expected_file_start is not None and row["index"] != expected_file_start + physical_row:
                        issues.add(
                            "rows.order",
                            relative,
                            None,
                            physical_row,
                            "Rows ordered by contiguous global index within their data file",
                            {
                                "expected_index": expected_file_start + physical_row,
                                "observed_index": repr(row["index"]),
                            },
                        )
                    ep_id = row["episode_index"]
                    if not _is_index(ep_id) or ep_id not in by_id:
                        issues.add(
                            "rows.episode",
                            relative,
                            None,
                            physical_row,
                            "A declared episode identifier",
                            repr(ep_id),
                        )
                        physical_row += 1
                        continue
                    episode = by_id[ep_id]
                    local_index = counts[ep_id]
                    counts[ep_id] += 1
                    if paths_by_id.get(ep_id) != path:
                        issues.add(
                            "rows.reference",
                            relative,
                            ep_id,
                            physical_row,
                            "Rows stored in their declared data file",
                            str(paths_by_id.get(ep_id)),
                        )
                    for column, expected in (
                        ("frame_index", local_index),
                        ("index", episode["dataset_from_index"] + local_index),
                    ):
                        if not _is_index(row[column]) or row[column] != expected:
                            issues.add(
                                f"rows.{column}", relative, ep_id, physical_row, expected, repr(row[column])
                            )
                    timestamp = row["timestamp"]
                    if (
                        isinstance(timestamp, bool)
                        or not isinstance(timestamp, (int, float))
                        or not math.isfinite(timestamp)
                    ):
                        issues.add(
                            "timestamps.finite",
                            relative,
                            ep_id,
                            physical_row,
                            "A finite numeric timestamp in seconds",
                            repr(timestamp),
                        )
                    else:
                        timestamp = float(timestamp)
                        if ep_id in previous_ts and timestamp <= previous_ts[ep_id]:
                            issues.add(
                                "timestamps.order",
                                relative,
                                ep_id,
                                physical_row,
                                "Strictly increasing episode timestamps",
                                {"previous_s": previous_ts[ep_id], "current_s": timestamp},
                            )
                        previous_ts[ep_id] = timestamp
                        expected_ts = local_index / info["fps"]
                        if abs(timestamp - expected_ts) > tolerance_s:
                            issues.add(
                                "timestamps.grid",
                                relative,
                                ep_id,
                                physical_row,
                                "timestamp = frame_index / fps within the effective tolerance",
                                {
                                    "expected_s": expected_ts,
                                    "observed_s": timestamp,
                                    "error_s": timestamp - expected_ts,
                                },
                            )
                    physical_row += 1
            complete_files += 1
            next_dataset_index += physical_row
        except (OSError, ValueError, TypeError, NotImplementedError) as exc:
            report.complete = False
            preceding_files_readable = False
            unavailable_episodes.update(ep_id for ep_id, expected in paths_by_id.items() if expected == path)
            report.add(
                "rows.readable",
                "verified_fault",
                "error",
                {"file": relative, "row": physical_row},
                "Readable parquet rows with required index and timestamp columns",
                str(exc),
                next_action="Inspect this local artifact; its remaining rows could not be checked.",
            )
    for ep_id, episode in by_id.items():
        if ep_id in unavailable_episodes:
            continue
        report.coverage["inspected_episodes"] += 1
        if counts[ep_id] != episode["length"]:
            report.add(
                "episodes.length",
                "verified_fault",
                "error",
                {"episode_index": ep_id, "file": paths_by_id.get(ep_id, root).relative_to(root).as_posix()},
                episode["length"],
                counts[ep_id],
                measurements={"units": "rows"},
                next_action="Compare the episode's declared length and range with its observed rows.",
            )
    if complete_files == len(data_paths) and report.coverage["inspected_rows"] != info["total_frames"]:
        report.add(
            "rows.totals",
            "verified_fault",
            "error",
            {"file": "meta/info.json"},
            info["total_frames"],
            report.coverage["inspected_rows"],
            measurements={"units": "rows"},
            next_action="Compare all locally stored data rows with the declared total_frames.",
        )
    if len(report.findings) == findings_before and report.complete:
        report.add(
            "episodes.coverage",
            "passed",
            "info",
            {},
            "Contiguous episode identifiers/ranges and matching row indices and lengths",
            "All inspected episode metadata and tabular rows satisfy these invariants",
            coverage=dict(report.coverage),
        )
        report.add(
            "timestamps.consistency",
            "passed",
            "info",
            {},
            "Finite, strictly increasing timestamps on the frame_index / fps grid",
            "All inspected rows satisfy the timing contract",
            measurements={"tolerance_s": tolerance_s, "fps": info["fps"]},
            coverage={"rows": report.coverage["inspected_rows"]},
            limits="Dataset timestamps do not establish physical camera synchronization.",
        )
    return episodes
