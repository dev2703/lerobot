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

"""Offline video reference checks with a disk-backed presentation timestamp index."""

import logging
import math
import random
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _av_available, _pyarrow_available, require_package

from .models import AuditReport

if TYPE_CHECKING or _av_available:
    import av

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
_BATCH_SIZE = 4096
_SAMPLE_ROWS = 64


@dataclass
class _Reference:
    episode: dict[str, Any]
    camera: str
    start: float
    end: float
    feature: dict[str, Any]

    @property
    def location(self) -> dict[str, Any]:
        return {
            "episode": self.episode["episode_index"],
            "camera": self.camera,
            "video_interval_s": [self.start, self.end],
        }


def _template_path(root: Path, template: str, **values: Any) -> Path:
    relative = Path(template.format(**values))
    # Cache snapshots legitimately contain symlinks to sibling blobs. Validate the
    # metadata's lexical path rather than resolving those read-only symlinks.
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact template must be relative and stay within the dataset")
    return root / relative


def _dimensions(feature: dict[str, Any]) -> tuple[int, int] | None:
    shape = feature.get("shape")
    names = feature.get("names")
    if not isinstance(shape, (list, tuple)) or not isinstance(names, (list, tuple)):
        return None
    if len(shape) != len(names) or "height" not in names or "width" not in names:
        return None
    height, width = shape[names.index("height")], shape[names.index("width")]
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        return None
    return height, width


def _episode_rows(root: Path, info: dict[str, Any], episode: dict[str, Any]) -> Iterator[tuple[int, float]]:
    path = _template_path(
        root,
        info["data_path"],
        chunk_index=episode["data/chunk_index"],
        file_index=episode["data/file_index"],
    )
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(
            batch_size=_BATCH_SIZE, columns=["episode_index", "index", "timestamp"]
        ):
            for row in batch.to_pylist():
                if row["episode_index"] != episode["episode_index"]:
                    continue
                timestamp = row["timestamp"]
                if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                    raise ValueError("Observation timestamp is not numeric")
                if not math.isfinite(timestamp):
                    raise ValueError("Observation timestamp is not finite")
                yield row["index"], float(timestamp)


def _queries(
    root: Path, info: dict[str, Any], reference: _Reference, mode: str, seed: int
) -> Iterator[tuple[int, float]]:
    rows = _episode_rows(root, info, reference.episode)
    if mode == "full":
        yield from rows
        return
    # Reservoir sampling avoids reading all episode rows into memory. A stable
    # textual seed avoids Python's randomized hash and keeps selection portable.
    rng = random.Random(f"{seed}:{reference.camera}:{reference.episode['episode_index']}")
    reservoir: list[tuple[int, float]] = []
    for count, row in enumerate(rows):
        if count < _SAMPLE_ROWS:
            reservoir.append(row)
        else:
            slot = rng.randrange(count + 1)
            if slot < _SAMPLE_ROWS:
                reservoir[slot] = row
    yield from sorted(reservoir)


def _scan_pts(path: Path, database: sqlite3.Connection, report: AuditReport) -> dict[str, Any]:
    """Decode once, retaining timestamps on disk and only one decoded frame in RAM."""
    database.execute("CREATE TABLE frames (pts REAL NOT NULL)")
    count = 0
    missing_pts = 0
    unordered_pts = 0
    previous: float | None = None
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    last_duration: float | None = None
    stream_end: float | None = None
    dimensions: set[tuple[int, int]] = set()
    # A local filename can contain a playlist referencing remote resources.
    # Restrict libavformat itself, including C-level nested I/O, to local files.
    with av.open(str(path), mode="r", options={"protocol_whitelist": "file"}) as container:
        if not container.streams.video:
            raise ValueError("Artifact contains no video stream")
        stream = container.streams.video[0]
        if stream.codec_context is None:
            raise ValueError("Video stream lacks a valid codec context")
        if stream.duration is not None and stream.time_base is not None:
            stream_end = float(((stream.start_time or 0) + stream.duration) * stream.time_base)
        # Explicitly opening the codec distinguishes an unsupported execution
        # dependency from corrupt packets encountered during decoding.
        stream.codec_context.open()
        for frame in container.decode(stream):
            count += 1
            report.coverage["decoded_video_frames"] = report.coverage.get("decoded_video_frames", 0) + 1
            dimensions.add((frame.height, frame.width))
            if frame.pts is None or stream.time_base is None:
                missing_pts += 1
                continue
            timestamp = float(frame.pts * stream.time_base)
            if not math.isfinite(timestamp):
                missing_pts += 1
                continue
            if previous is not None and timestamp <= previous:
                unordered_pts += 1
            previous = timestamp
            first_timestamp = timestamp if first_timestamp is None else min(first_timestamp, timestamp)
            if last_timestamp is None or timestamp >= last_timestamp:
                last_timestamp = timestamp
                duration = getattr(frame, "duration", 0)
                last_duration = (
                    float(duration * frame.time_base)
                    if duration and duration > 0 and frame.time_base is not None
                    else None
                )
            database.execute("INSERT INTO frames VALUES (?)", (timestamp,))
            if count % _BATCH_SIZE == 0:
                database.commit()
                logger.info("Decoded %s frames from %s", count, path.name)
    database.commit()
    database.execute("CREATE INDEX frame_pts ON frames(pts)")
    return {
        "decoded_frames": count,
        "missing_pts": missing_pts,
        "nonincreasing_pts": unordered_pts,
        "dimensions": sorted(dimensions),
        "first_timestamp_s": first_timestamp,
        "last_timestamp_s": last_timestamp,
        "decoded_end_s": (
            last_timestamp + last_duration
            if last_timestamp is not None and last_duration is not None
            else None
        ),
        "stream_end_s": stream_end,
    }


def _nearest(database: sqlite3.Connection, timestamp: float) -> float | None:
    earlier = database.execute(
        "SELECT pts FROM frames WHERE pts <= ? ORDER BY pts DESC LIMIT 1", (timestamp,)
    ).fetchone()
    later = database.execute(
        "SELECT pts FROM frames WHERE pts >= ? ORDER BY pts ASC LIMIT 1", (timestamp,)
    ).fetchone()
    candidates = [abs(row[0] - timestamp) for row in (earlier, later) if row is not None]
    return min(candidates) if candidates else None


def _check_reference(
    root: Path,
    info: dict[str, Any],
    path: Path,
    reference: _Reference,
    database: sqlite3.Connection,
    scan: dict[str, Any],
    report: AuditReport,
    video_scan: str,
    sample_seed: int,
    tolerance_s: float,
) -> None:
    location = {"file": str(path.relative_to(root)), **reference.location}
    known_ends = [scan[key] for key in ("decoded_end_s", "stream_end_s") if scan[key] is not None]
    end = min(known_ends) if known_ends else None
    start = scan["first_timestamp_s"]
    if start is None or end is None:
        report.complete = False
        report.add(
            "video.interval_bounds",
            "not_checked",
            "warning",
            location,
            "Known decoded-stream start and end covering the entire referenced interval",
            {"start_s": start, "end_s": end},
            next_action="Provide a decodable stream with presentation timestamps and duration metadata.",
        )
    else:
        outside = reference.start < start - tolerance_s or reference.end > end + tolerance_s
        report.add(
            "video.interval_bounds",
            "verified_fault" if outside else "passed",
            "error" if outside else "info",
            location,
            "The complete declared video interval lies inside the available decoded-stream bounds",
            {"available_start_s": start, "available_end_s": end},
            measurements={"tolerance_s": tolerance_s},
            coverage={"mode": "full file decode"},
            limits="End uses the final decoded frame's duration and container stream duration where available.",
            next_action="Review this episode's video offset and end timestamp before training."
            if outside
            else "",
        )
    expected_dimensions = _dimensions(reference.feature)
    if expected_dimensions is None:
        report.complete = False
        report.add(
            "video.dimensions",
            "not_checked",
            "warning",
            location,
            "A positive height and width declared in feature shape/names",
            reference.feature.get("shape"),
            next_action="Check the video feature's shape and dimension names.",
        )
    elif any(tuple(shape) != expected_dimensions for shape in scan["dimensions"]):
        report.add(
            "video.dimensions",
            "verified_fault",
            "error",
            location,
            list(expected_dimensions),
            scan["dimensions"],
            next_action="Review feature dimensions and the referenced video before training.",
        )
    inspected, unmatched, outside = 0, 0, 0
    max_distance = 0.0
    first_bad: list[dict[str, Any]] = []
    try:
        for row_index, timestamp in _queries(root, info, reference, video_scan, sample_seed):
            requested = reference.start + timestamp
            inspected += 1
            report.coverage["video_observations_checked"] = (
                report.coverage.get("video_observations_checked", 0) + 1
            )
            # End is exclusive in metadata; allow only floating point noise,
            # not the lookup tolerance, to move a query onto the next episode.
            out_of_interval = timestamp < -tolerance_s or requested >= reference.end + 1e-9
            outside += int(out_of_interval)
            distance = _nearest(database, requested)
            if distance is not None:
                max_distance = max(max_distance, distance)
            bad = distance is None or distance > tolerance_s
            unmatched += int(bad)
            if (bad or out_of_interval) and len(first_bad) < 5:
                first_bad.append(
                    {"row": row_index, "requested_timestamp_s": requested, "distance_s": distance}
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.complete = False
        report.add(
            "video.observation_queries",
            "not_checked",
            "error",
            location,
            "Readable numeric observation timestamps",
            str(error),
            coverage={"inspected_observations": inspected},
            next_action="Resolve the data-table finding, then repeat the video lookup check.",
        )
        return
    if inspected == 0:
        report.complete = False
        report.add(
            "video.observation_queries",
            "not_checked",
            "error",
            location,
            "At least one referenced observation",
            0,
            next_action="Resolve missing episode rows before checking video coverage.",
        )
        return
    measurements = {
        "unmatched_observations": unmatched,
        "observations_outside_interval": outside,
        "maximum_nearest_frame_distance_s": max_distance,
        "tolerance_s": tolerance_s,
        "first_affected_rows": first_bad,
    }
    report.add(
        "video.observation_coverage",
        "verified_fault" if unmatched or outside else "passed",
        "error" if unmatched or outside else "info",
        location,
        "Every inspected shifted timestamp is inside its episode interval and within tolerance of a decoded PTS",
        measurements,
        measurements=measurements,
        coverage={
            "mode": video_scan,
            "inspected_observations": inspected,
            "declared_observations": reference.episode["length"],
        },
        limits="Nearest-PTS lookup follows the PyAV reader; camera capture synchronization is not established.",
        next_action="Review offsets and affected frame timestamps before training."
        if unmatched or outside
        else "",
    )
    report.coverage["video_intervals_checked"] = report.coverage.get("video_intervals_checked", 0) + 1


def audit_videos(
    root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    report: AuditReport,
    *,
    video_scan: str = "full",
    sample_seed: int = 0,
    tolerance_s: float = 1e-4,
) -> None:
    """Check local v3.0 video references, decoding each distinct file only once."""
    features = info.get("features", {})
    cameras = {key: feature for key, feature in features.items() if feature.get("dtype") == "video"}
    images = [key for key, feature in features.items() if feature.get("dtype") == "image"]
    report.coverage["video_cameras"] = len(cameras)
    report.coverage["video_lookup_mode"] = video_scan
    report.coverage["video_decode_scope"] = "all frames in referenced files"
    if images:
        report.add(
            "video.image_features",
            "not_checked",
            "info",
            {"features": images},
            "Video-dtype camera features",
            "Image-only camera features are outside the initial video audit",
            next_action="Use a separate image-integrity or visual-review workflow if needed.",
        )
    if not cameras:
        report.add("video.references", "passed", "info", {}, "No video references to inspect", 0)
        return
    try:
        require_package("av", "dataset")
        require_package("pyarrow", "dataset")
        if not _av_available or not _pyarrow_available:
            raise ImportError("PyAV and PyArrow are required for video checks")
    except ImportError as error:
        report.complete = False
        report.add(
            "video.dependencies",
            "not_checked",
            "error",
            {},
            "An available PyAV decoder and PyArrow reader",
            str(error),
            next_action="Install the dataset extra in the execution environment, then repeat the audit.",
        )
        return
    references: dict[Path, list[_Reference]] = defaultdict(list)
    for episode in episodes:
        for key, feature in cameras.items():
            location = {"episode": episode["episode_index"], "camera": key}
            try:
                prefix = f"videos/{key}/"
                chunk, file_index = episode[prefix + "chunk_index"], episode[prefix + "file_index"]
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (chunk, file_index)
                ):
                    raise ValueError("Video chunk/file indices must be nonnegative integers")
                raw_start, raw_end = episode[prefix + "from_timestamp"], episode[prefix + "to_timestamp"]
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in (raw_start, raw_end)
                ):
                    raise ValueError("Video interval endpoints must be numeric")
                start, end = float(raw_start), float(raw_end)
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise ValueError("Video interval must be finite, nonnegative, and increasing")
                path = _template_path(
                    root, info["video_path"], video_key=key, chunk_index=chunk, file_index=file_index
                )
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                report.add(
                    "video.reference_metadata",
                    "verified_fault",
                    "error",
                    location,
                    "A relative video path and valid chunk/file indices and timestamp interval",
                    str(error),
                    next_action="Review this episode's video metadata before training.",
                )
                continue
            references[path].append(_Reference(episode, key, start, end, feature))
    report.coverage["referenced_video_files"] = len(references)
    report.coverage["referenced_video_intervals"] = sum(map(len, references.values()))
    for path, refs in sorted(references.items()):
        location = {
            "file": str(path.relative_to(root)),
            "episodes": sorted({reference.episode["episode_index"] for reference in refs}),
        }
        for camera in sorted({reference.camera for reference in refs}):
            previous_reference: _Reference | None = None
            for reference in sorted((ref for ref in refs if ref.camera == camera), key=lambda ref: ref.start):
                if previous_reference is not None and reference.start < previous_reference.end - tolerance_s:
                    report.add(
                        "video.interval_overlap",
                        "verified_fault",
                        "error",
                        {"file": location["file"], **reference.location},
                        "Distinct episodes have nonoverlapping intervals in a shared camera video",
                        {
                            "previous_episode": previous_reference.episode["episode_index"],
                            "previous_interval_s": [previous_reference.start, previous_reference.end],
                        },
                        next_action="Review shared-video episode offsets before training.",
                    )
                if previous_reference is None or reference.end > previous_reference.end:
                    previous_reference = reference
        if not path.is_file():
            report.complete = False
            report.add(
                "video.file_exists",
                "verified_fault",
                "error",
                location,
                "A readable local referenced video file",
                "Missing or broken local reference",
                limits="A missing cached artifact does not establish corruption of the remote dataset.",
                next_action="Supply the missing local artifact and repeat the offline audit.",
            )
            continue
        logger.info("Auditing video %s (%s episode references)", path.name, len(refs))
        with (
            tempfile.TemporaryDirectory(prefix="lerobot-audit-pts-") as temp_dir,
            closing(sqlite3.connect(Path(temp_dir) / "pts.sqlite")) as database,
        ):
            try:
                scan = _scan_pts(path, database, report)
            except av.error.DecoderNotFoundError as error:
                report.complete = False
                report.add(
                    "video.decoder",
                    "not_checked",
                    "error",
                    location,
                    "A supported decoder for this video's codec",
                    str(error),
                    next_action="Use a PyAV/FFmpeg installation supporting this codec, then repeat the audit.",
                )
                continue
            except (av.error.FFmpegError, OSError, ValueError) as error:
                report.add(
                    "video.decode",
                    "verified_fault",
                    "error",
                    location,
                    "A readable video stream with decodable packets",
                    str(error),
                    next_action="Review or replace the unreadable local artifact before training.",
                )
                continue
            report.coverage["video_files_decoded"] = report.coverage.get("video_files_decoded", 0) + 1
            failed = scan["decoded_frames"] == 0
            unusual_pts = scan["missing_pts"] or scan["nonincreasing_pts"]
            report.add(
                "video.presentation_timestamps",
                "verified_fault" if failed else "warning" if unusual_pts else "passed",
                "error" if failed else "warning" if unusual_pts else "info",
                location,
                "A nonempty decoded stream with finite strictly increasing presentation timestamps",
                scan,
                measurements=scan,
                coverage={"mode": "full file decode", "episode_references": len(refs)},
                limits="Unusual PTS alone does not prove a reader lookup fails; requested coverage is checked separately.",
                next_action="Review the local video stream before training." if failed or unusual_pts else "",
            )
            for reference in refs:
                _check_reference(
                    root,
                    info,
                    path,
                    reference,
                    database,
                    scan,
                    report,
                    video_scan,
                    sample_seed,
                    tolerance_s,
                )
