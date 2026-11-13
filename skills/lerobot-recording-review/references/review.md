# Offline recording review

Run from the repository root with an existing environment containing NumPy,
PyArrow, and PyAV (the dataset extra). No dependency installation or Hub access
occurs. `--root` may point directly to an already cached dataset or Hub snapshot.
Resolve ambiguous caches explicitly; never silently select a different revision.

```bash
uv run --no-sync --offline python \
  skills/lerobot-recording-review/scripts/review_recordings.py \
  --root /path/to/cached/dataset \
  --output-dir /private/tmp/recording-review \
  --max-episodes 3 --samples-per-episode 5 --seed 17
```

Repeat `--expected-camera observation.images.KEY` for cameras the user says should
exist. Camera roles come from the user or recording configuration. Neither
`iphone` nor `wrist` proves a camera is wrist-mounted.

Use `--export-frames` when visual inspection is authorized and useful. This writes
private sampled PNG files alongside the report; it does not upload them. Output must
be a new directory outside both the dataset and repository. Defaults export no
frames and copy no task strings. Reports should also stay private. Exit `0`
means numeric sampling completed within its stated scope, with pending semantic
review; `2` means incomplete evidence, unsupported format, missing dependencies,
or interruption. Neither status certifies demonstration quality.

## Sampling and source contract

The helper supports v3.0 Parquet episode metadata and RGB video features only.
It selects first and last episodes plus a seeded subset of the middle when the
budget permits. One episode uses its temporal midpoint; two use the endpoints;
larger time budgets distribute integer frame indices across the episode.
Explicitly report small budgets: they can miss the entire grasp or contact phase.

The relevant contracts are in `src/lerobot/datasets/dataset_metadata.py`
(`get_video_file_path`), `dataset_reader.py` (video timestamp shifting), and
`video_utils.py` (nearest presentation timestamp). Several episodes can reference
one video. Episode-relative times are shifted by each camera's
`videos/<key>/from_timestamp`; `to_timestamp` bounds the interval. The helper
seeks to individual samples, checking nearest PTS within `1e-4` seconds. It does
not scan all video frames, check row integrity, or certify reader compatibility.
An unavailable sample may be a local cache omission or decoding/metadata issue;
route structural questions to an audit rather than declaring remote corruption.
Image-only and depth features remain explicitly uninspected.

Depth markers take precedence over camera names and apparent channel count. The
helper recognizes `info.is_depth_map`, `info["video.is_depth_map"]`, and
`video_info["video.is_depth_map"]`, using the truthiness semantics in
`src/lerobot/configs/video.py`. For example, a nonempty string `"false"` is not a
JSON boolean `false`; it causes the stream to be excluded like upstream. Do not
silently reinterpret it or rewrite metadata during review. Malformed feature
objects are reported unavailable while other supported cameras can still run.

An absent depth marker does not prove a stream is RGB. Colorized depth can have
three channels, and a grayscale stream could be infrared rather than depth. If
recording/conversion provenance suggests missing modality tags, verify the source
configuration before interpreting RGB heuristics. Raw 12/16-bit depth codes and
physical depth units are not brightness measurements. Do not infer modality from
names, shapes, or a plausible-looking preview alone; preserve ambiguity when the
producer's modality declaration cannot be established.

Measurements use 64x64 grayscale previews in 0–255 intensity units. Defaults flag
mean below 12 (dark), spatial standard deviation below 4 (low contrast), or mean
absolute difference below 1 between adjacent sampled previews (little change).
They are transparent screening heuristics, not calibrated quality thresholds.
Little change between distant stills cannot establish continuous idleness;
movement may have occurred between them. Spatial contrast cannot locate an object
or measure focus reliably. Numeric sampling always leaves task visibility pending.

## Camera and task-phase review

| Evidence                                                  | Review question                                                                                               | Legitimate alternative / next check                                                                                                        |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Wrist view misses the object during approach              | Does the task require visual alignment before contact? Compare the same episode times in the overview camera. | Object may enter the wrist view only close to contact. Inspect approach-to-contact video before flagging persistent loss.                  |
| Wrist view loses the object at contact or transport       | Is the gripper blocking the useful view? Is the target still observable in another view?                      | Occlusion inside a successful grasp or looking toward the destination can be expected. Do not call this a failed grasp from absence alone. |
| Wrist view changes but never shows the relevant workspace | Across approach, grasp, transfer, and release, does it supply information needed for this task?               | Normal brightness/motion cannot answer this. Mark task visibility unassessed until actual frames/clips and camera roles are inspected.     |
| Dark/flat frames or a stationary view                     | Is this sensor obstruction, poor exposure, a static scene, or intentional waiting?                            | Compare adjacent frames, other cameras, and action/state traces if available. Sparse stills do not prove a freeze or idle duration.        |
| Apparent failed grasp or inconsistent approach            | Is the object visibly secured and does the supplied success criterion hold?                                   | Recovery demonstrations and alternative successful strategies can be valuable. Inspect the sequence and retain ambiguity.                  |
| Similar starting frames                                   | Are object pose, workspace, lighting, and approach conditions actually varied?                                | A fixed camera/background or a tiny sample is insufficient to conclude low dataset diversity.                                              |

Assign phases only when observed; uniform sample positions are not task labels.
Compare episode-relative times across views without claiming physical sensor
synchronization. If only one camera is available, state the missing corroboration.
Record inspected intervals and frame timestamps separately; viewing stills covers
those stills, not the intervals between them.

## Queue and decisions

Keep each candidate's stable `check_id`, episode, camera, interval in seconds,
supporting measurements/frame references, classification, qualitative confidence,
alternatives, and next inspection. Automatic candidates start `pending`. Record
human responses in a separate `decisions.json`, for example:

```json
[
  {
    "candidate_id": "e0-c0-visibility",
    "disposition": "rejected",
    "note": "Wrist occlusion is expected during this successful grasp."
  }
]
```

Only record an actual human's decision; do not fill in approval on their behalf.
Report accepted and rejected flags among adjudicated candidates, pending count,
and the small sample size. Human acceptance measures usefulness of flags, not
ground-truth model accuracy. When comparing behavior with and without the skill,
use the same recordings and record unsupported claims, interventions, elapsed
time, and whether camera/phase uncertainty was preserved. Do not infer a broad
quality improvement from a few test cases.
