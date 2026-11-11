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

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.audit import tables as audit_tables_module
from lerobot.datasets.audit.models import AuditReport
from lerobot.datasets.audit.tables import audit_tables

pytestmark = pytest.mark.dataset


def _fixture(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    info = {
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 6,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    episodes = [
        {
            "episode_index": ep_id,
            "length": 3,
            "dataset_from_index": ep_id * 3,
            "dataset_to_index": ep_id * 3 + 3,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for ep_id in range(2)
    ]
    rows = [
        {"index": i, "episode_index": i // 3, "frame_index": i % 3, "timestamp": (i % 3) / 30}
        for i in range(6)
    ]
    _write(root, episodes, rows)
    return info, episodes, rows


def _write(root: Path, episodes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    metadata = root / "meta/episodes/chunk-000/file-000.parquet"
    data = root / "data/chunk-000/file-000.parquet"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episodes), metadata)
    pq.write_table(pa.Table.from_pylist(rows), data)


def _ids(report: AuditReport) -> set[str]:
    return {finding.check_id for finding in report.findings if finding.classification == "verified_fault"}


def test_valid_shared_data_file_and_input_preserved(tmp_path: Path) -> None:
    info, episodes, _ = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    report = AuditReport()
    assert audit_tables(tmp_path, info, report, 1e-4) == episodes
    assert report.complete
    assert not _ids(report)
    assert report.coverage["inspected_rows"] == 6
    assert report.coverage["inspected_episodes"] == 2
    assert report.coverage["data_files"] == 1
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_episode_length_mismatch(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    episodes[0]["length"] = 4
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert {"episodes.ranges", "episodes.length"} <= _ids(report)
    assert report.complete


@pytest.mark.parametrize("start,end", [(2, 5), (4, 3), (4, 7)])
def test_overlapping_invalid_or_gapped_ranges(tmp_path: Path, start: int, end: int) -> None:
    info, episodes, rows = _fixture(tmp_path)
    episodes[1]["dataset_from_index"] = start
    episodes[1]["dataset_to_index"] = end
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert "episodes.ranges" in _ids(report)


@pytest.mark.parametrize("timestamp", [0.0, -0.1])
def test_duplicate_and_backward_timestamps(tmp_path: Path, timestamp: float) -> None:
    info, episodes, rows = _fixture(tmp_path)
    rows[1]["timestamp"] = timestamp
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert {"timestamps.order", "timestamps.grid"} <= _ids(report)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), None])
def test_nonfinite_or_null_timestamp(tmp_path: Path, timestamp: float | None) -> None:
    info, episodes, rows = _fixture(tmp_path)
    rows[1]["timestamp"] = timestamp
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert "timestamps.finite" in _ids(report)


def test_timestamp_tolerance_override(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    rows[1]["timestamp"] += 0.0005
    _write(tmp_path, episodes, rows)
    strict = AuditReport()
    audit_tables(tmp_path, info, strict, 1e-4)
    relaxed = AuditReport()
    audit_tables(tmp_path, info, relaxed, 1e-3)
    assert "timestamps.grid" in _ids(strict)
    assert "timestamps.grid" not in _ids(relaxed)


def test_duplicate_rows_and_frame_indices(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    rows[1] = dict(rows[0])
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert {"rows.index", "rows.frame_index", "rows.order"} <= _ids(report)


def test_interleaved_episodes_fail_physical_row_order(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    rows[:] = [rows[i] for i in (0, 3, 1, 4, 2, 5)]
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert "rows.order" in _ids(report)


def test_sorted_shard_order_must_match_global_row_order(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    episodes[0]["data/file_index"] = 1
    _write(tmp_path, episodes, rows[3:])
    pq.write_table(pa.Table.from_pylist(rows[:3]), tmp_path / "data/chunk-000/file-001.parquet")
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert "rows.file_order" in _ids(report)


def test_episode_metadata_physical_order_matches_identifier(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    _write(tmp_path, list(reversed(episodes)), rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert "episodes.order" in _ids(report)
    assert report.complete


def test_episode_discovery_matches_reader_layout(tmp_path: Path) -> None:
    info, _, _ = _fixture(tmp_path)
    metadata = tmp_path / "meta/episodes/chunk-000/file-000.parquet"
    unsupported = metadata.parent / "deeper/file-000.parquet"
    unsupported.parent.mkdir()
    metadata.rename(unsupported)
    report = AuditReport()
    assert audit_tables(tmp_path, info, report, 1e-4) == []
    assert not report.complete


def test_unknown_episode_row(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    rows[1]["episode_index"] = 20
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert "rows.episode" in _ids(report)


def test_missing_referenced_data_is_incomplete(tmp_path: Path) -> None:
    info, _, _ = _fixture(tmp_path)
    (tmp_path / "data/chunk-000/file-000.parquet").unlink()
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert not report.complete
    assert any(finding.check_id == "rows.reference" for finding in report.findings)


def test_partial_episode_cache_is_incomplete(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    _write(tmp_path, episodes[:1], rows[:3])
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert not report.complete
    assert any(finding.check_id == "episodes.available" for finding in report.findings)


def test_corrupt_parquet_retains_partial_findings(tmp_path: Path) -> None:
    info, _, _ = _fixture(tmp_path)
    (tmp_path / "data/chunk-000/file-000.parquet").write_bytes(b"not parquet")
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert not report.complete
    assert "rows.readable" in _ids(report)


def test_missing_required_column(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    for row in rows:
        del row["frame_index"]
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert not report.complete
    assert "rows.readable" in _ids(report)


def test_invalid_metadata_index_is_a_finding(tmp_path: Path) -> None:
    info, episodes, rows = _fixture(tmp_path)
    episodes[0]["data/file_index"] = -1
    _write(tmp_path, episodes, rows)
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert "episodes.metadata" in _ids(report)


def test_invalid_template_is_reported(tmp_path: Path) -> None:
    info, _, _ = _fixture(tmp_path)
    info["data_path"] = "../outside.parquet"
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert not report.complete
    assert "rows.reference" in _ids(report)


def test_unexpected_reader_visible_parquet_is_audited(tmp_path: Path) -> None:
    info, _, _ = _fixture(tmp_path)
    data = tmp_path / "data/chunk-000/file-000.parquet"
    (data.parent / "backup.parquet").write_bytes(data.read_bytes())
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert report.coverage["data_files"] == 2
    assert report.coverage["inspected_rows"] == 12
    assert {"rows.reference", "rows.totals", "episodes.length"} <= _ids(report)
    assert report.exit_code == 1


def test_custom_data_layout_outside_reader_discovery_is_incomplete(tmp_path: Path) -> None:
    info, _, _ = _fixture(tmp_path)
    destination = tmp_path / "custom/chunk-000/file-000.parquet"
    destination.parent.mkdir(parents=True)
    (tmp_path / "data/chunk-000/file-000.parquet").rename(destination)
    info["data_path"] = "custom/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    report = AuditReport()
    audit_tables(tmp_path, info, report, 1e-4)
    assert not report.complete
    assert report.exit_code == 2
    assert any(finding.check_id == "rows.layout" for finding in report.findings)


def test_interrupted_read_retains_discovered_row_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info, episodes, rows = _fixture(tmp_path)
    rows[0]["frame_index"] = 10
    _write(tmp_path, episodes, rows)
    real_parquet_file = pq.ParquetFile

    class InterruptedParquet:
        def __init__(self, path: Path) -> None:
            self.parquet = real_parquet_file(path)
            self.schema_arrow = self.parquet.schema_arrow

        def iter_batches(self, **kwargs: Any) -> Any:
            for batch in self.parquet.iter_batches(**kwargs):
                yield batch.slice(0, 2)
                raise KeyboardInterrupt

    def parquet_file(path: Path) -> Any:
        if path.is_relative_to(tmp_path / "data"):
            return InterruptedParquet(path)
        return real_parquet_file(path)

    monkeypatch.setattr(audit_tables_module.pq, "ParquetFile", parquet_file)
    report = AuditReport()
    with pytest.raises(KeyboardInterrupt):
        audit_tables(tmp_path, info, report, 1e-4)
    assert report.coverage["inspected_rows"] == 2
    assert "rows.frame_index" in _ids(report)
