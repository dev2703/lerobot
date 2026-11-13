#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Generate a private, offline, sampled recording-review queue without importing LeRobot."""

import argparse
import importlib.metadata
import json
import math
import random
from collections.abc import Iterator
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    av: Any = importlib.import_module("av")
    np: Any = importlib.import_module("numpy")
    pq: Any = importlib.import_module("pyarrow.parquet")
except ImportError:
    av = np = pq = None

TOLERANCE_S = 1e-4


def is_depth_map(feature: dict[str, Any] | None) -> bool:
    """Mirror configs/video.py depth markers without importing the LeRobot package."""
    feature = feature or {}
    info = feature.get("info") or {}
    video_info = feature.get("video_info")
    return bool(
        info.get("is_depth_map")
        or info.get("video.is_depth_map")
        or (isinstance(video_info, dict) and video_info.get("video.is_depth_map"))
    )


def local_path(root: Path, relative: str) -> Path:
    """Allow cache blob symlinks, but reject absolute and traversing metadata paths."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Metadata path must be relative without '..' components")
    return root / path


def episode_sample(indices: list[int], budget: int, seed: int) -> list[int]:
    """Select reproducible episodes, preserving temporal endpoints when possible."""
    ordered = sorted(indices)
    if len(ordered) <= budget:
        return ordered
    if budget == 1:
        return [random.Random(seed).choice(ordered)]
    middle = random.Random(seed).sample(ordered[1:-1], budget - 2)
    return sorted([ordered[0], *middle, ordered[-1]])


def sample_indices(length: int, budget: int) -> list[int]:
    """Choose distinct integer frame indices across the episode."""
    count = min(length, budget)
    if count == 1:
        return [(length - 1) // 2]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def read_previews(
    path: Path, targets: list[float], start: float, end: float
) -> Iterator[tuple[Any, Any, float]]:
    """Reuse one local-only decoder for all requested stills from this interval."""
    # Python socket mocks cannot stop FFmpeg's native or nested protocol I/O.
    with av.open(str(path), mode="r", options={"protocol_whitelist": "file"}) as container:
        stream = container.streams.video[0]
        for target in targets:
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            best, best_pts, distance = None, None, math.inf
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                pts = float(frame.pts * frame.time_base)
                if start - TOLERANCE_S <= pts < end and abs(pts - target) < distance:
                    best, best_pts, distance = frame, pts, abs(pts - target)
                if pts >= target:
                    break
            if best is None or best_pts is None or distance > TOLERANCE_S:
                raise ValueError("No sampled frame within the recorded PTS tolerance")
            gray = best.reformat(width=64, height=64, format="gray").to_ndarray().astype("float32")
            yield gray, best, best_pts


def inspect_clip(
    path: Path,
    start: float,
    end: float,
    offset: float,
    fps: float,
    max_frames: int,
    export_path: Path | None = None,
) -> dict[str, Any]:
    """Decode a contiguous bounded window, counting keyframe pre-roll in the budget."""
    result: dict[str, Any] = {"frames": [], "decoded_frames": 0, "complete": False}
    with ExitStack() as stack:
        container = stack.enter_context(av.open(str(path), mode="r", options={"protocol_whitelist": "file"}))
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        output = encoder = None
        first_pts = None
        last_frame_end = None
        if export_path is not None:
            output = stack.enter_context(av.open(str(export_path), mode="w"))
            encoder = output.add_stream("mpeg4", rate=Fraction(str(fps)).limit_denominator(100000))
            encoder.width, encoder.height = stream.codec_context.width, stream.codec_context.height
            encoder.pix_fmt = "yuv420p"
            encoder.codec_context.time_base = Fraction(1, 60000)
        for frame in container.decode(stream):
            result["decoded_frames"] += 1
            if frame.pts is not None and frame.time_base is not None:
                pts = float(frame.pts * frame.time_base)
                if pts >= end - TOLERANCE_S:
                    result["complete"] = True
                    break
                if start - TOLERANCE_S <= pts < end:
                    duration = getattr(frame, "duration", 0)
                    tail_duration = (
                        float(duration * frame.time_base)
                        if duration and duration > 0
                        else float(1 / stream.average_rate)
                        if stream.average_rate
                        else None
                    )
                    last_frame_end = pts + tail_duration if tail_duration is not None else None
                    gray = frame.reformat(width=64, height=64, format="gray").to_ndarray()
                    result["frames"].append(
                        {
                            "episode_time_s": pts - offset,
                            "video_pts_s": pts,
                            "gray_mean": float(gray.mean()),
                            "gray_std": float(gray.std()),
                        }
                    )
                    if output is not None and encoder is not None:
                        first_pts = pts if first_pts is None else first_pts
                        exported = av.VideoFrame.from_ndarray(
                            frame.to_ndarray(format="rgb24"), format="rgb24"
                        )
                        exported.pts = round((pts - first_pts) * 60000)
                        exported.time_base = Fraction(1, 60000)
                        for packet in encoder.encode(exported):
                            output.mux(packet)
            if result["decoded_frames"] >= max_frames:
                break
        else:
            # Reaching EOF only covers the request if its final frame was present.
            result["complete"] = last_frame_end is not None and last_frame_end >= end - TOLERANCE_S
        if output is not None and encoder is not None and result["frames"]:
            for packet in encoder.encode():
                output.mux(packet)
    return result


def candidate(
    episode: int,
    camera: str,
    camera_index: int,
    check: str,
    times: list[float],
    evidence: dict[str, Any],
    alternatives: str,
) -> dict[str, Any]:
    """Describe a review candidate without converting it to a task-success label."""
    return {
        "candidate_id": f"e{episode}-c{camera_index}-{check}",
        "check_id": f"recording.{check}",
        "episode_index": episode,
        "camera": camera,
        "interval_s": [min(times), max(times)] if times else None,
        "classification": "not_checked" if check in {"visibility", "unavailable"} else "hypothesis",
        "severity": "info" if check == "visibility" else "warning",
        "reason": {
            "visibility": "Task-relevant visual coverage requires inspection of the actual scene and task phase.",
            "unavailable": "The requested local camera samples could not all be inspected.",
            "dark": "Sampled grayscale intensity is below the screening threshold.",
            "low-contrast": "Sampled spatial contrast is below the screening threshold.",
        }.get(check, "Adjacent sampled stills have little pixel change; intervening motion is unknown."),
        "confidence": "unassessed" if check in {"visibility", "unavailable"} else "low",
        "evidence": evidence,
        "alternative_interpretations": alternatives,
        "next_action": "Inspect the relevant task phase and compare other available camera views.",
        "disposition": "pending",
    }


def review(
    root: Path,
    *,
    max_episodes: int = 3,
    samples_per_episode: int = 5,
    seed: int = 17,
    expected_cameras: tuple[str, ...] = (),
    export_dir: Path | None = None,
    clip_requests: tuple[tuple[int, str, float], ...] = (),
    clip_radius_s: float = 1,
    max_clip_seconds: float = 5,
    max_clip_frames: int = 300,
    clip_export_dir: Path | None = None,
) -> dict[str, Any]:
    """Inspect local metadata and sampled frames; preserve partial evidence on failure."""
    root = root.expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "incomplete",
        "source_root": str(root),
        "settings": {
            "max_episodes": max_episodes,
            "samples_per_episode": samples_per_episode,
            "seed": seed,
            "pts_tolerance_s": TOLERANCE_S,
            "dark_mean_below": 12,
            "low_contrast_std_below": 4,
            "little_change_mad_below": 1,
            "clip_radius_s": clip_radius_s if math.isfinite(clip_radius_s) else None,
            "max_clip_seconds": max_clip_seconds if math.isfinite(max_clip_seconds) else None,
            "max_clip_frames": max_clip_frames,
        },
        "coverage": {
            "selected_episodes": [],
            "declared_episodes": None,
            "sampled_frames": 0,
            "requested_frames": 0,
            "semantic_reviewed_frames": 0,
            "full_video_scan": False,
            "structural_audit": "not_checked",
            "requested_clips": len(clip_requests),
            "clip_frames": 0,
            "clip_decoded_frames": 0,
            "inspected_clip_intervals": [],
        },
        "samples": [],
        "clips": [],
        "queue": [],
        "unavailable": [],
        "limits": "Numeric samples cannot establish task visibility, grasp success, continuous idleness, "
        "physical synchronization, or quality of unsampled frames. Human review remains pending.",
    }
    try:
        if max_episodes < 1 or samples_per_episode < 1:
            raise ValueError("Sample budgets must be positive")
        if (
            len(clip_requests) > 10
            or not math.isfinite(clip_radius_s)
            or not math.isfinite(max_clip_seconds)
            or clip_radius_s <= 0
            or not 0 < max_clip_seconds <= 30
            or 2 * clip_radius_s > max_clip_seconds
            or not 1 <= max_clip_frames <= 3000
        ):
            raise ValueError(
                "Clips require <=10 windows, positive radius, <=30 seconds and <=3000 decode frames"
            )
        for directory in (export_dir, clip_export_dir):
            if directory is not None:
                directory = directory.resolve()
                if directory == root or directory.is_relative_to(root) or root.is_relative_to(directory):
                    raise ValueError("Preview output must be outside the dataset")
                directory.mkdir(parents=True, exist_ok=False)
        if av is None or np is None or pq is None:
            raise RuntimeError(
                "Required dependencies unavailable: NumPy, PyArrow, PyAV; no installation attempted"
            )
        report["dependencies"] = {
            name: importlib.metadata.version(name) for name in ("av", "numpy", "pyarrow")
        }
        info = json.loads((root / "meta/info.json").read_text())
        if not isinstance(info, dict):
            raise ValueError("Dataset metadata must be a JSON object")
        report["format"] = info.get("codebase_version")
        storage_format = info.get("storage_format") or "lerobot"
        if info.get("codebase_version") != "v3.0" or storage_format != "lerobot":
            raise ValueError("Only v3.0 lerobot storage with Parquet episode metadata is supported")
        fps = float(info["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and positive")
        features = info["features"]
        if not isinstance(features, dict):
            raise ValueError("Dataset features must be a JSON object")
        cameras = []
        for key, feature in features.items():
            if not isinstance(feature, dict):
                report["unavailable"].append(f"{key}: feature metadata must be a JSON object")
                continue
            if feature.get("info") is not None and not isinstance(feature["info"], dict):
                report["unavailable"].append(f"{key}: feature info must be a JSON object or null")
                continue
            if feature.get("dtype") == "image" or is_depth_map(feature):
                report["unavailable"].append(f"{key}: image-only/depth inspection is unsupported")
            elif feature.get("dtype") == "video":
                cameras.append(key)
        cameras.sort()
        report["cameras"] = cameras
        for key in sorted(set(expected_cameras) - set(cameras)):
            report["unavailable"].append(f"Expected camera unavailable: {key}")
        if not cameras:
            raise ValueError("No supported RGB video cameras available")
        columns = ["episode_index", "length"]
        for key in cameras:
            columns.extend(
                f"videos/{key}/{field}"
                for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            )
        episodes: dict[int, dict[str, Any]] = {}
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=256):
                for row in batch.to_pylist():
                    index = row["episode_index"]
                    if not isinstance(index, int) or index < 0 or index in episodes:
                        raise ValueError(
                            "Episode metadata contains invalid/duplicate episode IDs; run an audit"
                        )
                    episodes[index] = row
        if not episodes:
            raise ValueError("No local episode metadata available")
        report["coverage"]["declared_episodes"] = info.get("total_episodes")
        report["coverage"]["available_episodes"] = len(episodes)
        if len(episodes) != info.get("total_episodes"):
            report["unavailable"].append(
                "Local episode metadata count differs from declared total; run an audit"
            )
        selected = episode_sample(list(episodes), max_episodes, seed)
        report["coverage"]["selected_episodes"] = selected
        for episode in selected:
            row = episodes[episode]
            length = row["length"]
            if not isinstance(length, int) or length < 1:
                raise ValueError(f"Invalid episode length for episode {episode}; run an audit")
            times = [i / fps for i in sample_indices(length, samples_per_episode)]
            for camera_index, camera in enumerate(cameras):
                report["coverage"]["requested_frames"] += len(times)
                previews, observed = [], []
                try:
                    prefix = f"videos/{camera}"
                    start, end = float(row[f"{prefix}/from_timestamp"]), float(row[f"{prefix}/to_timestamp"])
                    if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
                        raise ValueError("Invalid source-video interval; run an audit")
                    if start + times[-1] >= end + TOLERANCE_S:
                        raise ValueError(
                            "Requested sample extends beyond source-video interval; run an audit"
                        )
                    relative = info["video_path"].format(
                        video_key=camera,
                        chunk_index=row[f"{prefix}/chunk_index"],
                        file_index=row[f"{prefix}/file_index"],
                    )
                    path = local_path(root, relative)
                    frames = read_previews(path, [start + time for time in times], start, end)
                    for sample_index, (time, (gray, frame, pts)) in enumerate(
                        zip(times, frames, strict=True)
                    ):
                        sample = {
                            "episode_index": episode,
                            "camera": camera,
                            "episode_time_s": time,
                            "video_offset_s": start,
                            "video_pts_s": pts,
                            "video_path": relative,
                            "width": frame.width,
                            "height": frame.height,
                            "gray_mean": float(gray.mean()),
                            "gray_std": float(gray.std()),
                        }
                        if export_dir is not None:
                            filename = f"e{episode}-c{camera_index}-s{sample_index}.png"
                            frame.to_image().save(export_dir / filename)
                            sample["frame_path"] = f"frames/{filename}"
                        report["samples"].append(sample)
                        report["coverage"]["sampled_frames"] += 1
                        previews.append(gray)
                        observed.append(sample)
                except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                    report["unavailable"].append(f"Episode {episode}, camera {camera}: {type(exc).__name__}")
                    report["queue"].append(
                        candidate(
                            episode,
                            camera,
                            camera_index,
                            "unavailable",
                            times,
                            {"error_type": type(exc).__name__},
                            "Local cache omission, unsupported codec, or invalid metadata; remote corruption is unproven.",
                        )
                    )
                report["queue"].append(
                    candidate(
                        episode,
                        camera,
                        camera_index,
                        "visibility",
                        times,
                        {"sample_times_s": [s["episode_time_s"] for s in observed]},
                        "Wrist occlusion, off-camera transit, and release may be legitimate. Normal pixel metrics "
                        "also occur when the camera never captures the relevant workspace.",
                    )
                )
                for check, metric, threshold in (("dark", "gray_mean", 12), ("low-contrast", "gray_std", 4)):
                    flagged = [s for s in observed if s[metric] < threshold]
                    if flagged:
                        report["queue"].append(
                            candidate(
                                episode,
                                camera,
                                camera_index,
                                check,
                                [s["episode_time_s"] for s in flagged],
                                {
                                    "metric": metric,
                                    "threshold_below": threshold,
                                    "values": [s[metric] for s in flagged],
                                    "units": "gray_0_255",
                                },
                                "Exposure, a uniform background, or task-phase occlusion may be legitimate.",
                            )
                        )
                for i in range(1, len(previews)):
                    difference = float(np.abs(previews[i] - previews[i - 1]).mean())
                    if difference < 1:
                        entry = candidate(
                            episode,
                            camera,
                            camera_index,
                            f"little-change-{i}",
                            [observed[i - 1]["episode_time_s"], observed[i]["episode_time_s"]],
                            {
                                "mean_absolute_difference": difference,
                                "threshold_below": 1,
                                "units": "gray_0_255",
                            },
                            "Intentional waiting, a fixed view, or movement between sampled stills; "
                            "continuous idleness and a frozen sensor are unverified.",
                        )
                        entry["check_id"] = "recording.little-change"
                        report["queue"].append(entry)
        for request_index, (episode, camera, center) in enumerate(clip_requests):
            clip: dict[str, Any] = {
                "clip_id": f"clip-{request_index}",
                "episode_index": episode,
                "camera": camera,
                "center_s": center if math.isfinite(center) else None,
                "status": "unavailable",
                "complete": False,
                "frames": [],
            }
            report["clips"].append(clip)
            try:
                if episode not in episodes or camera not in cameras:
                    raise ValueError("Clip episode or RGB camera is unavailable")
                row = episodes[episode]
                length = row["length"]
                if not isinstance(length, int) or length < 1:
                    raise ValueError("Invalid clip episode length")
                if not math.isfinite(center) or not 0 <= center < length / fps:
                    raise ValueError("Clip center must lie inside the episode")
                prefix = f"videos/{camera}"
                offset = float(row[f"{prefix}/from_timestamp"])
                source_end = float(row[f"{prefix}/to_timestamp"])
                if not all(math.isfinite(t) for t in (offset, source_end)) or not 0 <= offset < source_end:
                    raise ValueError("Invalid clip source-video interval")
                start = max(0, center - clip_radius_s)
                end = min(length / fps, source_end - offset, center + clip_radius_s)
                if end <= start:
                    raise ValueError("Clip has no source-video interval")
                relative = info["video_path"].format(
                    video_key=camera,
                    chunk_index=row[f"{prefix}/chunk_index"],
                    file_index=row[f"{prefix}/file_index"],
                )
                clip.update(requested_interval_s=[start, end], video_offset_s=offset, video_path=relative)
                filename = f"{clip['clip_id']}.mp4"
                inspection = inspect_clip(
                    local_path(root, relative),
                    offset + start,
                    offset + end,
                    offset,
                    fps,
                    max_clip_frames,
                    clip_export_dir / filename if clip_export_dir is not None else None,
                )
                clip.update(inspection)
                report["coverage"]["clip_frames"] += len(clip["frames"])
                report["coverage"]["clip_decoded_frames"] += clip["decoded_frames"]
                if not clip["frames"]:
                    raise ValueError("No frames decoded within the requested clip interval and decode budget")
                clip["status"] = "complete" if clip["complete"] else "partial"
                clip["actual_interval_s"] = [
                    clip["frames"][0]["episode_time_s"],
                    clip["frames"][-1]["episode_time_s"],
                ]
                if clip_export_dir is not None:
                    clip["clip_path"] = f"clips/{filename}"
                report["coverage"]["inspected_clip_intervals"].append(
                    {
                        "clip_id": clip["clip_id"],
                        "episode_index": episode,
                        "camera": camera,
                        "interval_s": clip["actual_interval_s"],
                        "semantic_review": "pending",
                    }
                )
                if not clip["complete"]:
                    report["unavailable"].append(
                        f"{clip['clip_id']}: decode budget or source EOF prevented complete clip coverage"
                    )
                for check, metric, threshold in (("dark", "gray_mean", 12), ("low-contrast", "gray_std", 4)):
                    flagged = [frame for frame in clip["frames"] if frame[metric] < threshold]
                    if flagged:
                        entry = candidate(
                            episode,
                            camera,
                            cameras.index(camera),
                            f"{check}-{clip['clip_id']}",
                            [frame["episode_time_s"] for frame in flagged],
                            {
                                "clip_id": clip["clip_id"],
                                "metric": metric,
                                "threshold_below": threshold,
                                "values": [frame[metric] for frame in flagged],
                                "units": "gray_0_255",
                            },
                            "Brief occlusion, exposure changes, or uniform backgrounds may be legitimate; "
                            "inspect the clip and other cameras before judging task success.",
                        )
                        entry["check_id"] = f"recording.{check}"
                        report["queue"].append(entry)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                clip["error"] = f"{type(exc).__name__}: {exc}"
                report["unavailable"].append(f"{clip['clip_id']}: {clip['error']}")
        report["status"] = "incomplete" if report["unavailable"] else "sampled"
    except KeyboardInterrupt:
        report["unavailable"].append("Sampling interrupted; retained evidence is partial")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report["unavailable"].append(f"{type(exc).__name__}: {exc}")
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    """Write numeric findings and pending review candidates outside source data."""
    (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Recording review",
        "",
        f"Status: {report['status']}. Human review pending.",
        "",
        report["limits"],
        "",
        "Coverage: " + json.dumps(report["coverage"]),
        "",
        "## Review queue",
        "",
    ]
    for item in report["queue"]:
        # JSON quoting keeps dataset-controlled camera keys from creating Markdown links/HTML.
        location = json.dumps({k: item[k] for k in ("episode_index", "camera", "interval_s")})
        lines.extend(
            [
                f"### {item['candidate_id']} ({item['check_id']})",
                "",
                "```json",
                location,
                "```",
                "",
                f"Classification: {item['classification']}; confidence: {item['confidence']}; "
                f"disposition: {item['disposition']}.",
                "",
                "```json",
                json.dumps(item["evidence"]),
                "```",
                "",
                item["alternative_interpretations"],
                "",
                item["next_action"],
                "",
            ]
        )
    lines.extend(["## Targeted clips", "", "Numeric inspection only; semantic review remains pending.", ""])
    for clip in report.get("clips", []):
        lines.extend(["```json", json.dumps(clip), "```", ""])
    lines.extend(["## Unavailable evidence", "", "```json", json.dumps(report["unavailable"]), "```", ""])
    (output / "review.md").write_text("\n".join(lines))


def main() -> int:
    """Run a bounded local inspection; never overwrite an existing report directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expected-camera", action="append", default=[])
    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument(
        "--clip", nargs=3, action="append", default=[], metavar=("EPISODE", "CAMERA", "CENTER_SECONDS")
    )
    parser.add_argument("--clip-radius-s", type=float, default=1)
    parser.add_argument("--max-clip-seconds", type=float, default=5)
    parser.add_argument("--max-clip-frames", type=int, default=300)
    parser.add_argument("--export-clips", action="store_true")
    args = parser.parse_args()
    try:
        clip_requests = tuple((int(episode), camera, float(center)) for episode, camera, center in args.clip)
    except ValueError:
        parser.error("Clip episode and center must be an integer and a number respectively")
    if args.export_clips and not clip_requests:
        parser.error("--export-clips requires at least one --clip")
    root, output = args.root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (root, repository)):
        parser.error("Output must be outside the dataset and repository")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"Cannot create a fresh output directory: {exc}")
    report = review(
        root,
        max_episodes=args.max_episodes,
        samples_per_episode=args.samples_per_episode,
        seed=args.seed,
        expected_cameras=tuple(args.expected_camera),
        export_dir=output / "frames" if args.export_frames else None,
        clip_requests=clip_requests,
        clip_radius_s=args.clip_radius_s,
        max_clip_seconds=args.max_clip_seconds,
        max_clip_frames=args.max_clip_frames,
        clip_export_dir=output / "clips" if args.export_clips else None,
    )
    write_report(report, output)
    return 0 if report["status"] == "sampled" else 2


if __name__ == "__main__":
    raise SystemExit(main())
