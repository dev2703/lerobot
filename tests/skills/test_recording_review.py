# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Exercise the skill's standalone sampler on small synthetic v3.0 recordings."""

import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = pytest.mark.dataset

SCRIPT = Path(__file__).resolve().parents[2] / "skills/lerobot-recording-review/scripts/review_recordings.py"
spec = importlib.util.spec_from_file_location("recording_review", SCRIPT)
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)
CAMERA = "observation.images.wrist"


def make_dataset(root: Path, mode: str = "moving") -> Path:
    """Write two episodes sharing one video, with independent episode-local times."""
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 4,
        "total_episodes": 2,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {CAMERA: {"dtype": "video", "shape": [32, 32, 3]}},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = [
        {
            "episode_index": e,
            "length": 8,
            f"videos/{CAMERA}/chunk_index": 0,
            f"videos/{CAMERA}/file_index": 0,
            f"videos/{CAMERA}/from_timestamp": float(e * 2),
            f"videos/{CAMERA}/to_timestamp": float((e + 1) * 2),
        }
        for e in range(2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
    video = root / f"videos/{CAMERA}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    rng = np.random.default_rng(42)
    fixed = rng.integers(32, 220, (32, 32, 3), dtype="uint8")
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(16):
            image = rng.integers(32, 220, (32, 32, 3), dtype="uint8")
            if mode == "frozen":
                image = fixed
            elif mode == "dark-first" and index < 8 or mode == "brief-occlusion" and index == 3:
                image = np.zeros((32, 32, 3), dtype="uint8")
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return root


def rewrite_metadata(root: Path, **changes: Any) -> None:
    """Change only synthetic metadata to inject a known fault."""
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0].update(changes)
    pq.write_table(pa.Table.from_pylist(rows), path)


def hashes(root: Path) -> dict[str, str]:
    """Fingerprint all source bytes to detect unintended writes."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_shared_video_offsets_and_unassessed_semantics(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "dataset", "dark-first")
    report = reviewer.review(root, samples_per_episode=3)
    assert report["status"] == "sampled"
    assert report["coverage"]["sampled_frames"] == 6
    assert report["coverage"]["semantic_reviewed_frames"] == 0
    assert report["coverage"]["full_video_scan"] is False
    assert [s["video_pts_s"] for s in report["samples"]] == [0, 1, 1.75, 2, 3, 3.75]
    dark = [q for q in report["queue"] if q["check_id"] == "recording.dark"]
    assert {q["episode_index"] for q in dark} == {0}
    assert all(q["classification"] == "hypothesis" and q["disposition"] == "pending" for q in dark)


def test_no_network_or_input_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_dataset(tmp_path / "dataset")
    before = hashes(root)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network access attempted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    report = reviewer.review(root)
    output = tmp_path / "report"
    output.mkdir()
    reviewer.write_report(report, output)
    assert report["status"] == "sampled"
    assert hashes(root) == before
    assert json.loads((output / "review.json").read_text())["samples"] == report["samples"]
    assert not list(output.rglob("*.png"))


@pytest.mark.parametrize("mode", ["moving", "frozen", "brief-occlusion"])
def test_pixel_metrics_never_decide_wrist_task_visibility(tmp_path: Path, mode: str) -> None:
    # A moving view of irrelevant workspace and a useful view can have identical statistics.
    # Intentional waiting and transient grasp occlusion must not become verified faults either.
    report = reviewer.review(make_dataset(tmp_path / "dataset", mode), samples_per_episode=8)
    visibility = [q for q in report["queue"] if q["check_id"] == "recording.visibility"]
    assert len(visibility) == 2
    assert all(q["classification"] == "not_checked" and q["confidence"] == "unassessed" for q in visibility)
    assert all(q["disposition"] == "pending" for q in report["queue"])
    assert {q["classification"] for q in report["queue"]} <= {"hypothesis", "not_checked"}
    if mode == "brief-occlusion":
        dark = [q for q in report["queue"] if q["check_id"] == "recording.dark"]
        assert len(dark) == 1
        assert dark[0]["interval_s"] == [0.75, 0.75]
    if mode == "frozen":
        assert any(q["check_id"] == "recording.little-change" for q in report["queue"])


def test_reproducible_episode_and_sparse_time_sampling() -> None:
    result = reviewer.episode_sample(list(range(20)), 5, 17)
    assert result == reviewer.episode_sample(list(reversed(range(20))), 5, 17)
    assert len(result) == 5 and result[0] == 0 and result[-1] == 19
    assert reviewer.sample_indices(8, 1) == [3]
    assert reviewer.sample_indices(8, 2) == [0, 7]
    assert reviewer.sample_indices(2, 5) == [0, 1]


def test_missing_expected_camera_is_incomplete(tmp_path: Path) -> None:
    report = reviewer.review(make_dataset(tmp_path / "dataset"), expected_cameras=("observation.images.top",))
    assert report["status"] == "incomplete"
    assert report["unavailable"] and report["coverage"]["sampled_frames"] == 10


@pytest.mark.parametrize("fault", ["missing-video", "corrupt-video", "offset", "interval", "duplicate-id"])
def test_unavailable_samples_do_not_pass(tmp_path: Path, fault: str) -> None:
    root = make_dataset(tmp_path / "dataset")
    video = next(root.rglob("*.mp4"))
    if fault == "missing-video":
        video.unlink()
    elif fault == "corrupt-video":
        video.write_bytes(b"not a video")
    elif fault == "offset":
        rewrite_metadata(root, **{f"videos/{CAMERA}/from_timestamp": 0.02})
    elif fault == "interval":
        rewrite_metadata(root, **{f"videos/{CAMERA}/to_timestamp": 0.1})
    elif fault == "duplicate-id":
        rewrite_metadata(root, episode_index=1)
    report = reviewer.review(root)
    assert report["status"] == "incomplete"
    assert report["unavailable"]
    assert report["coverage"]["semantic_reviewed_frames"] == 0


def test_missing_decoder_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_dataset(tmp_path / "dataset")
    monkeypatch.setattr(reviewer, "av", None)
    report = reviewer.review(root)
    assert report["status"] == "incomplete"
    assert report["coverage"]["sampled_frames"] == 0


def test_interruption_keeps_partial_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_dataset(tmp_path / "dataset")
    original = reviewer.read_preview
    calls = 0

    def interrupted(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args)

    monkeypatch.setattr(reviewer, "read_preview", interrupted)
    report = reviewer.review(root)
    assert report["status"] == "incomplete"
    assert report["coverage"]["sampled_frames"] == 1
    assert len(report["samples"]) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"codebase_version": "v2.1"},
        {"storage_format": "lance"},
        {"fps": 0},
        {"fps": float("nan")},
        {"total_episodes": 3},
    ],
)
def test_unsupported_or_incomplete_metadata(tmp_path: Path, changes: dict[str, Any]) -> None:
    root = make_dataset(tmp_path / "dataset")
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info.update(changes)
    path.write_text(json.dumps(info))
    report = reviewer.review(root)
    assert report["status"] == "incomplete"
    assert report["unavailable"]


def test_image_only_camera_cannot_receive_video_pass(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "dataset")
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"][CAMERA]["dtype"] = "image"
    path.write_text(json.dumps(info))
    report = reviewer.review(root)
    assert report["status"] == "incomplete"
    assert report["coverage"]["sampled_frames"] == 0


def test_private_frame_export(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "dataset")
    before = hashes(root)
    output = tmp_path / "private-review"
    output.mkdir()
    report = reviewer.review(root, samples_per_episode=1, export_dir=output / "frames")
    assert report["status"] == "sampled"
    assert len(list((output / "frames").glob("*.png"))) == 2
    assert all((output / s["frame_path"]).is_file() for s in report["samples"])
    assert hashes(root) == before


@pytest.mark.parametrize("destination", ["dataset", "repository", "existing"])
def test_cli_rejects_unsafe_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    root = make_dataset(tmp_path / "dataset")
    output = {"dataset": root / "reports", "repository": SCRIPT.parent / "reports", "existing": tmp_path}[
        destination
    ]
    before = hashes(root)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(root), "--output-dir", str(output)])
    with pytest.raises(SystemExit) as error:
        reviewer.main()
    assert error.value.code == 2
    assert hashes(root) == before


def test_cli_success_and_incomplete_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_dataset(tmp_path / "dataset")
    for missing, expected in ((False, 0), (True, 2)):
        args = [str(SCRIPT), "--root", str(root), "--output-dir", str(tmp_path / f"report-{missing}")]
        if missing:
            args += ["--expected-camera", "absent"]
        monkeypatch.setattr(sys, "argv", args)
        assert reviewer.main() == expected


@pytest.mark.parametrize("budgets", [{"max_episodes": 0}, {"samples_per_episode": -1}])
def test_invalid_budgets(tmp_path: Path, budgets: dict[str, int]) -> None:
    report = reviewer.review(make_dataset(tmp_path / "dataset"), **budgets)
    assert report["status"] == "incomplete"
    assert not report["samples"]


@pytest.mark.parametrize("path", ["../outside.mp4", "/outside.mp4"])
def test_metadata_path_escape(path: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        reviewer.local_path(tmp_path, path)
