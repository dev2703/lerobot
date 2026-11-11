# Running and interpreting an offline dataset audit

This pilot supports the default Parquet/video backend of the `v3.0` format declared by
[`dataset_metadata.py`](../../../src/lerobot/datasets/dataset_metadata.py).
Its reusable implementation lives in
[`lerobot.datasets.audit`](../../../src/lerobot/datasets/audit/).
Inspect the installed implementation when using another checkout or release.
Alternative `storage_format` backends, such as `lance`, return an incomplete
report; they are outside this pilot.

## Invocation

Choose exactly one source. Keep the output directory outside the dataset and
its source cache snapshot:

```bash
lerobot-audit-dataset \
  --root /path/to/dataset \
  --output-dir /path/to/reports/dataset-audit

lerobot-audit-dataset \
  --repo-id owner/dataset \
  --revision v3.0 \
  --output-dir /path/to/reports/dataset-audit
```

From a source environment, use
`uv run --no-sync --offline lerobot-audit-dataset` after the command has been
installed in that environment. This prevents environment synchronization during
an offline inspection. Inspect `--help` for the available interface. If
dependencies are unavailable, report what is missing; installation is outside an
audit request.

`--repo-id` resolves existing local LeRobot or Hugging Face cache entries. It
does not authenticate or download. `--revision` selects a locally cached revision.
When the resolver lists ambiguous candidates, select an explicit revision or
use `--root` with the intended local directory; do not silently choose a
different dataset.

Video scanning defaults to `--video-scan full`. In `sample` mode, seeded
reservoir sampling selects up to 64 observation timestamp lookups per episode
and camera. Both modes still decode all frames in every referenced video file;
sampling reduces lookup coverage, not decoding work. For example:

```bash
lerobot-audit-dataset \
  --root /path/to/dataset \
  --video-scan sample \
  --sample-seed 17 \
  --output-dir /path/to/reports/dataset-sample
```

`--timestamp-tolerance-s` overrides the reader's default `1e-4` seconds. Record
why an override is justified; increasing tolerance does not repair timestamps
or prove camera synchronization.

For Python use:

```python
from lerobot.datasets.audit import audit_dataset

report = audit_dataset(root="/path/to/dataset")
report.write("/path/to/reports/dataset-audit")
status = report.exit_code
```

## Source invariants

| Check                         | Contract and evidence                                                                                                                                                                                                                                                                                                                | Interpretation limit                                                                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Episode boundaries and rows   | `dataset_metadata.py` and `dataset_writer.py` maintain episode IDs, declared lengths, contiguous half-open global ranges `[dataset_from_index, dataset_to_index)`, and data chunk/file references. Data rows carry global `index`, episode-local `frame_index`, and `episode_index`.                                                 | Multiple episodes can share a Parquet file. A storage file is not an episode. Missing local files prevent complete coverage.                                                                                 |
| Timestamps                    | `dataset_writer.py` records episode-local frame timestamps; `LeRobotDataset` declares a default tolerance of `1e-4` seconds for the `1/fps` timing contract. Check finite timestamps, strict ordering, frame/FPS agreement, and unexpected gaps.                                                                                     | These timestamps alone do not establish independent sensor capture synchronization or action/state alignment.                                                                                                |
| Video references and coverage | `dataset_reader.py` shifts requested row timestamps by `videos/<camera>/from_timestamp`. `video_utils.py` compares requested timestamps with decoded presentation timestamps and accepts a closest-frame distance `<= tolerance_s`. Writer metadata records video chunk/file indices and `[from_timestamp, to_timestamp)` intervals. | Encoded frame counts need not equal dataset row counts. A usable nearest frame and valid referenced interval are the relevant evidence. Full decoding establishes decodability, not semantic visual quality. |

The checker streams data and decoded video; shared video paths are scanned once.
Inspect the report's actual row, file, camera, interval, and frame coverage before
describing the scope of a pass. Sampling and cancellation must remain visible in
the conclusion. Image-only features do not receive a video-integrity pass.

## Reports and exit status

`audit.json` uses a versioned report schema for tools. `audit.md` presents the
same findings for people. Reports identify the input, source implementation,
dataset format, effective settings, observed coverage, and completeness.

Each finding separates its classification from severity and records its stable
check ID, artifact location, expected and observed behavior, measurements,
coverage, interpretation limits, and next action. Use the classifications as
follows:

| Classification | Meaning                                                             |
| -------------- | ------------------------------------------------------------------- |
| Verified fault | Observed violation of a supported invariant.                        |
| Warning        | Suspicious observation that may be legitimate.                      |
| Hypothesis     | Possible explanation needing distinguishing evidence.               |
| Not checked    | Evidence, dependency, configuration, or scope prevented inspection. |
| Passed         | The stated check succeeded over its stated coverage.                |

| Exit code | Meaning                                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`       | Supported audit completed without verified faults within the selected scope. Warnings may remain.                                                             |
| `1`       | Audit completed and found verified faults.                                                                                                                    |
| `2`       | Audit is incomplete, including unsupported inputs, unresolved cache candidates, missing dependencies, or interruption. Retain any findings already collected. |

An incomplete report can contain verified faults; exit `2` takes precedence over
`1`. A missing cached video does not prove that the Hub copy is corrupt. If the
decoder is unavailable, report the video check as unavailable rather than
claiming that the video is unreadable.

Private validation should retain reports outside the repository. Commit small
synthetic regression fixtures for confirmed failures, without copying private
frames, task text, or recordings.
