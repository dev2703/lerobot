# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Synthetic video coverage cases; no network, downloads, or private recordings."""

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lerobot.datasets.audit import video
from lerobot.datasets.audit.models import AuditReport

av = pytest.importorskip("av")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = pytest.mark.dataset
CAMERA = "observation.images.camera"


def _fixture(
    root: Path, *, video_fps: int = 10, frames: int = 6, episode_count: int = 2, episode_length: int = 3
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    info = {
        "fps": 10,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            CAMERA: {"dtype": "video", "shape": [12, 16, 3], "names": ["height", "width", "channels"]}
        },
    }
    episodes = []
    rows = []
    duration = episode_length / 10
    for episode_index in range(episode_count):
        episodes.append(
            {
                "episode_index": episode_index,
                "length": episode_length,
                "dataset_from_index": episode_index * episode_length,
                "dataset_to_index": (episode_index + 1) * episode_length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{CAMERA}/chunk_index": 0,
                f"videos/{CAMERA}/file_index": 0,
                f"videos/{CAMERA}/from_timestamp": episode_index * duration,
                f"videos/{CAMERA}/to_timestamp": (episode_index + 1) * duration,
            }
        )
        rows.extend(
            {
                "episode_index": episode_index,
                "index": episode_index * episode_length + frame_index,
                "timestamp": frame_index / 10,
            }
            for frame_index in range(episode_length)
        )
    data = root / "data/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), data)
    path = root / f"videos/{CAMERA}/chunk-000/file-000.mp4"
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=video_fps)
        stream.width, stream.height = 16, 12
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = np.full((12, 16, 3), index * 10 % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return info, episodes, path


def test_valid_shared_video_scanned_once(tmp_path, monkeypatch):
    info, episodes, _ = _fixture(tmp_path)
    original = video._scan_pts
    calls = []

    def scan(path, database, report):
        calls.append(path)
        return original(path, database, report)

    monkeypatch.setattr(video, "_scan_pts", scan)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 0
    assert len(calls) == 1
    assert report.coverage["video_observations_checked"] == 6
    assert report.coverage["video_intervals_checked"] == 2
    assert report.coverage["decoded_video_frames"] == 6


def test_encoded_frame_count_may_exceed_dataset_rows(tmp_path):
    info, episodes, _ = _fixture(tmp_path, video_fps=20, frames=12)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 0
    assert report.coverage["decoded_video_frames"] == 12
    assert report.coverage["video_observations_checked"] == 6


def test_missing_local_video_is_partial(tmp_path):
    info, episodes, path = _fixture(tmp_path)
    path.unlink()
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 2
    assert any(
        f.check_id == "video.file_exists" and f.classification == "verified_fault" for f in report.findings
    )
    assert "remote" in report.findings[-1].limits


@pytest.mark.parametrize("contents", [b"not a video", b""])
def test_unreadable_video_is_verified_fault(tmp_path, contents):
    info, episodes, path = _fixture(tmp_path)
    path.write_bytes(contents)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 1
    assert any(f.check_id == "video.decode" and f.classification == "verified_fault" for f in report.findings)


def test_truncated_video_lacks_required_coverage(tmp_path):
    info, episodes, _ = _fixture(tmp_path, frames=3)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    faults = [
        f
        for f in report.findings
        if f.check_id == "video.observation_coverage" and f.classification == "verified_fault"
    ]
    assert report.exit_code == 1
    assert faults[0].location["episode"] == 1
    assert faults[0].measurements["unmatched_observations"] == 3


def test_byte_truncated_video_cannot_pass(tmp_path):
    info, episodes, path = _fixture(tmp_path)
    contents = path.read_bytes()
    path.write_bytes(contents[: len(contents) // 2])
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 1
    assert any(f.classification == "verified_fault" for f in report.findings)


def test_incorrect_offset_and_shared_interval_overlap(tmp_path):
    info, episodes, _ = _fixture(tmp_path)
    episodes[1][f"videos/{CAMERA}/from_timestamp"] = 0.15
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 1
    assert any(f.check_id == "video.interval_overlap" for f in report.findings)
    assert any(
        f.check_id == "video.observation_coverage" and f.classification == "verified_fault"
        for f in report.findings
    )


def test_interval_cannot_extend_past_decoded_video(tmp_path):
    info, episodes, _ = _fixture(tmp_path)
    episodes[-1][f"videos/{CAMERA}/to_timestamp"] = 100.0
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 1
    faults = [
        f
        for f in report.findings
        if f.check_id == "video.interval_bounds" and f.classification == "verified_fault"
    ]
    assert len(faults) == 1 and faults[0].location["episode"] == 1
    assert faults[0].observed["available_end_s"] == pytest.approx(0.6)


def test_dimension_mismatch(tmp_path):
    info, episodes, _ = _fixture(tmp_path)
    info["features"][CAMERA]["shape"] = [16, 12, 3]
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 1
    assert any(
        f.check_id == "video.dimensions" and f.classification == "verified_fault" for f in report.findings
    )


def test_missing_decoder_is_partial(tmp_path, monkeypatch):
    info, episodes, _ = _fixture(tmp_path)
    monkeypatch.setattr(video, "_av_available", False)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 2
    assert any(
        f.check_id == "video.dependencies" and f.classification == "not_checked" for f in report.findings
    )


def test_unsupported_codec_is_partial(tmp_path, monkeypatch):
    info, episodes, _ = _fixture(tmp_path)

    def unavailable(*args):
        raise av.error.DecoderNotFoundError(1, "Unavailable codec")

    monkeypatch.setattr(video, "_scan_pts", unavailable)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 2
    assert any(f.check_id == "video.decoder" for f in report.findings)


def test_interrupt_keeps_decode_progress(tmp_path, monkeypatch):
    info, episodes, _ = _fixture(tmp_path)
    report = AuditReport()

    def interrupted(path, database, output):
        output.coverage["decoded_video_frames"] = 2
        raise KeyboardInterrupt

    monkeypatch.setattr(video, "_scan_pts", interrupted)
    with pytest.raises(KeyboardInterrupt):
        video.audit_videos(tmp_path, info, episodes, report)
    assert report.coverage["decoded_video_frames"] == 2
    # The top-level audit catches interruption and marks the report partial.


def test_images_are_explicitly_outside_video_scope(tmp_path):
    report = AuditReport()
    video.audit_videos(tmp_path, {"features": {"image": {"dtype": "image"}}}, [], report)
    assert report.exit_code == 0
    assert any(
        f.check_id == "video.image_features" and f.classification == "not_checked" for f in report.findings
    )


def test_sample_reservoir_is_repeatable_and_bounded(tmp_path):
    info, episodes, _ = _fixture(tmp_path)
    rows = [{"episode_index": 0, "index": index, "timestamp": index / 10} for index in range(200)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "data/chunk-000/file-000.parquet")
    reference = video._Reference(episodes[0], CAMERA, 0, 20, info["features"][CAMERA])
    with sqlite3.connect(":memory:") as database:
        video._index_observations(
            tmp_path, info, {tmp_path / "video.mp4": [reference]}, database, AuditReport()
        )
        first = list(video._queries(database, reference, "sample", 37))
        assert len(first) == video._SAMPLE_ROWS
        assert first == list(video._queries(database, reference, "sample", 37))
        assert first != list(video._queries(database, reference, "sample", 38))


@pytest.mark.parametrize("mode", ["full", "sample"])
def test_shared_observation_shard_read_once_across_episodes_and_cameras(tmp_path, monkeypatch, mode):
    info, episodes, path = _fixture(tmp_path, frames=2000, episode_count=20, episode_length=100)
    second_camera = "observation.images.second"
    info["features"][second_camera] = info["features"][CAMERA].copy()
    info["video_path"] = path.relative_to(tmp_path).as_posix()
    for episode in episodes:
        for key, value in list(episode.items()):
            if key.startswith(f"videos/{CAMERA}/"):
                episode[key.replace(CAMERA, second_camera)] = value
    original = video.pq.ParquetFile
    reads = []

    def observed_parquet(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(video.pq, "ParquetFile", observed_parquet)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report, video_scan=mode)
    assert report.exit_code == 0
    assert len(reads) == 1
    assert report.coverage["video_observation_rows_read"] == 2000
    assert report.coverage["video_observations_checked"] == (4000 if mode == "full" else 2560)


def test_spot_check_bounds_decoding_and_never_reads_observation_rows(tmp_path, monkeypatch):
    info, episodes, _ = _fixture(tmp_path, frames=2000, episode_count=20, episode_length=100)

    def forbidden(*args, **kwargs):
        pytest.fail("Spot mode must not scan observation Parquet")

    monkeypatch.setattr(video.pq, "ParquetFile", forbidden)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report, video_scan="spot", sample_seed=17)
    assert report.exit_code == 0
    assert report.coverage["decoded_video_frames"] < 2000
    assert report.coverage["decoded_video_frames"] <= report.coverage["video_spot_decode_budget_per_file"]
    assert report.coverage["video_spot_intervals_checked"] == 3
    assert report.coverage["video_spot_probes"] == 9
    assert report.coverage["video_observation_rows_read"] == 0
    assert report.coverage["video_observations_checked"] == 0
    assert any(
        f.check_id == "video.full_stream_integrity" and f.classification == "not_checked"
        for f in report.findings
    )
    assert not any(
        f.check_id in ("video.observation_coverage", "video.presentation_timestamps")
        and f.classification == "passed"
        for f in report.findings
    )


def test_spot_budget_exhaustion_is_incomplete(tmp_path, monkeypatch):
    info, episodes, _ = _fixture(tmp_path, frames=200, episode_count=2, episode_length=100)
    monkeypatch.setattr(video, "_SPOT_DECODE_BUDGET", 1)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report, video_scan="spot")
    assert report.exit_code == 2
    assert report.coverage["decoded_video_frames"] <= 6
    assert any(
        f.check_id == "video.grid_spot_check" and f.classification == "not_checked" for f in report.findings
    )


def test_video_reference_path_cannot_escape_dataset(tmp_path):
    info, episodes, _ = _fixture(tmp_path)
    info["video_path"] = "../video.mp4"
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert report.exit_code == 1
    assert any(f.check_id == "video.reference_metadata" for f in report.findings)


def test_libavformat_network_protocols_are_disabled(tmp_path, monkeypatch):
    info, episodes, path = _fixture(tmp_path)
    # Libavformat recognizes content independently of the misleading .mp4 suffix.
    # Requiring file-only protocols prevents nested HTTP access from local lists.
    path.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://127.0.0.1:9/segment.ts\n#EXT-X-ENDLIST\n"
    )
    original_open = video.av.open
    options = []

    def observed_open(*args, **kwargs):
        options.append(kwargs.get("options"))
        return original_open(*args, **kwargs)

    monkeypatch.setattr(video.av, "open", observed_open)
    report = AuditReport()
    video.audit_videos(tmp_path, info, episodes, report)
    assert options == [{"protocol_whitelist": "file"}]
    assert report.exit_code == 1
