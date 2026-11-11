# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Synthetic video coverage cases; no network, downloads, or private recordings."""

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
    root: Path, *, video_fps: int = 10, frames: int = 6
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
    for episode_index in range(2):
        episodes.append(
            {
                "episode_index": episode_index,
                "length": 3,
                "dataset_from_index": episode_index * 3,
                "dataset_to_index": (episode_index + 1) * 3,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{CAMERA}/chunk_index": 0,
                f"videos/{CAMERA}/file_index": 0,
                f"videos/{CAMERA}/from_timestamp": episode_index * 0.3,
                f"videos/{CAMERA}/to_timestamp": (episode_index + 1) * 0.3,
            }
        )
        rows.extend(
            {
                "episode_index": episode_index,
                "index": episode_index * 3 + frame_index,
                "timestamp": frame_index / 10,
            }
            for frame_index in range(3)
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
    first = list(video._queries(tmp_path, info, reference, "sample", 37))
    assert len(first) == video._SAMPLE_ROWS
    assert first == list(video._queries(tmp_path, info, reference, "sample", 37))
    assert first != list(video._queries(tmp_path, info, reference, "sample", 38))


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
