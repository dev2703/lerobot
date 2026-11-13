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

The helper supports v3.0 `lerobot` storage with Parquet episode metadata and RGB
video features only. Missing, null, and empty `storage_format` declarations select
the upstream default, as does explicit `"lerobot"`. `"parquet"` is not a registered
storage backend; Lance and other backends remain unsupported by this helper.
It selects first and last episodes plus a seeded subset of the middle when the
budget permits. One episode uses its temporal midpoint; two use the endpoints;
larger time budgets distribute integer frame indices across the episode.
Explicitly report small budgets: they can miss the entire grasp or contact phase.
The default five stills can also miss a single-frame occlusion. Increase the still
budget for broader screening, then use a bounded clip for an uncertain event;
normal still metrics do not remove the need to inspect task-relevant phases.

The relevant contracts are in `src/lerobot/datasets/dataset_metadata.py`
(`get_video_file_path`), `dataset_reader.py` (video timestamp shifting), and
`video_utils.py` (nearest presentation timestamp). Several episodes can reference
one video. Episode-relative times are shifted by each camera's
`videos/<key>/from_timestamp`; `to_timestamp` bounds the interval. The helper
reuses one decoder per episode/camera and seeks to its samples, checking nearest
PTS within `1e-4` seconds. FFmpeg's native and nested input protocols are restricted
to local files, including when a misleadingly named local file contains a playlist.
It does
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

## Bounded event inspection

When sparse stills leave a phase uncertain, inspect the smallest useful window.
Repeat `--clip EPISODE CAMERA CENTER_SECONDS` for particular episode-relative
events, including episodes outside the sparse sampling selection. For example:

```bash
uv run --no-sync --offline python \
  skills/lerobot-recording-review/scripts/review_recordings.py \
  --root /path/to/cached/dataset \
  --output-dir /private/tmp/recording-event-review \
  --clip 0 observation.images.wrist 0.75 \
  --clip-radius-s 0.5 --max-clip-frames 100 --export-clips
```

This inspects consecutive decoded frames in the one-second window centered at
0.75 seconds and exports a private MP4 under `clips/`. It clips the request to the
episode and source-video boundaries and applies each camera's source offset.
No clips are exported without `--export-clips`. Clips contain video only, start
at a zero playback timestamp, and preserve source frame timing; the report keeps
the original source PTS and episode-relative time of every included frame.

Defaults allow up to ten windows, radius one second, five seconds per window,
and 300 decoded frames per window. `--max-clip-seconds` can raise the duration
ceiling up to 30 seconds; `--max-clip-frames` can raise the decode ceiling up to
3,000 frames. Keyframe pre-roll counts against the frame ceiling. A long GOP,
truncated source, or exhausted budget can leave a clip `partial` or `unavailable`;
the report stays incomplete rather than implying the requested window was covered.
Exact `frames` timestamps and `actual_interval_s` describe retained frames, with
the final endpoint identifying the last frame rather than unseen later time.

Clip brightness/contrast screening can reveal a brief event missed by sparse
stills, but the flag remains a hypothesis. Inspect the exported clip before
interpreting occlusion or task success. Numeric clip inspection does not increase
`semantic_reviewed_frames`; keep human observations and dispositions separate.

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
Load and validate those decisions with the standard-library-only helper:

```bash
uv run --no-sync --offline python \
  skills/lerobot-recording-review/scripts/review_decisions.py \
  --report /private/tmp/recording-review/review.json \
  --decisions /private/tmp/recording-review/decisions.json \
  --output-dir /private/tmp/recording-review-adjudicated
```

The output must be a fresh directory outside the repository, dataset, and input
report/decisions directories. The helper preserves source data and input reports,
writes derived `review.json` and `review.md`, and records source hashes. Relative
private frame/clip references remain relative to the original `artifact_root`;
media are not copied. Exit `2` rejects duplicate/unknown candidate IDs, unsupported
dispositions, missing human notes, invalid inspection references, and stale hashes.
Only `pending`, `accepted`, and `rejected` are accepted dispositions. Each file is
an authoritative snapshot: candidates omitted from it remain pending. Reapplying
the same decisions does not inflate counts.

To track actual visual inspection as well, use the object form. This is a schema
example; replace its entries only with actual human responses and inspections:

```json
{
  "schema_version": "1.0",
  "decisions": [
    {
      "candidate_id": "e0-c0-dark-clip-0",
      "disposition": "rejected",
      "note": "Reviewed the clip: expected transient grasp occlusion."
    }
  ],
  "reviewed_samples": [
    {
      "episode_index": 0,
      "camera": "observation.images.wrist",
      "episode_time_s": 0.0
    }
  ],
  "reviewed_clips": ["clip-0"]
}
```

Sample coordinates must exactly match `samples` in the input report. Clips must
be completely exported and identified by their report `clip_id`; partial clips
cannot be credited as completely reviewed. Optional `report_sha256` pins a
decision snapshot to the exact input report bytes. `semantic_reviewed_frames`
counts explicitly reviewed sampled stills only. Reviewed clips and their actual
frame intervals are recorded separately; neither numeric decoding nor an
accepted/rejected disposition adds semantic coverage by itself. The helper
validates references and records supplied human attestations; it cannot verify
that a person actually viewed the media.

`decisions_summary` reports accepted, rejected, adjudicated, and pending counts
for all queue candidates and separately for heuristic flags (`hypothesis`). The
acceptance fraction is accepted / (accepted + rejected), or null when none have
been adjudicated. Report the small sample size. Human acceptance measures
usefulness of flags, not ground-truth model accuracy.

For paired agent trials, see the [evaluation tooling PR](https://github.com/dev2703/lerobot/pull/3).
When comparing behavior with and without the skill,
use the same recordings and record unsupported claims, interventions, elapsed
time, and whether camera/phase uncertainty was preserved. Do not infer a broad
quality improvement from a few test cases.
