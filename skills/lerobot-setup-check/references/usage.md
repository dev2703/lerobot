# Offline SO-101 setup inspection

The checker runs with LeRobot's base dependencies. Hardware extras, serial drivers,
OpenCV and the RealSense SDK are not required for file inspection. It does not
open ports or cameras, enumerate cameras, import hardware implementations, read
motor registers or invoke cleanup methods on devices.

## Input and commands

Use a native JSON run configuration containing `robot` and optionally `teleop`.
An existing recording configuration can be supplied: other top-level settings
are outside the check and are not copied into the report. YAML, arbitrary plugins
and bimanual setups are outside this pilot.

The current SO implementations share config classes between SO-100 and SO-101;
native serialization can emit `so100_follower` / `so100_leader` even for an SO-101
configuration. These aliases are accepted with an explicit warning. Neither name
establishes which physical arm is attached.

For example, save this as `so101.json` and substitute the intended paths and IDs:

```json
{
  "robot": {
    "type": "so101_follower",
    "port": "/dev/ttyACM0",
    "id": "my_follower",
    "cameras": {
      "front": {
        "type": "opencv",
        "index_or_path": 0,
        "width": 640,
        "height": 480,
        "fps": 30
      }
    }
  },
  "teleop": {
    "type": "so101_leader",
    "port": "/dev/ttyACM1",
    "id": "my_leader"
  }
}
```

```bash
uv run lerobot-check-setup --config so101.json --output-dir outputs/setup-inspection
```

When using an existing environment without refreshing installed entry points:

```bash
uv run --no-sync python -m lerobot.scripts.lerobot_check_setup \
  --config so101.json --output-dir outputs/setup-inspection
```

Relative paths resolve against the current working directory, as they do for the
runtime configuration. Explicit paths are used literally: a `~` in a JSON path
is not shell-expanded. An explicit `calibration_dir` is the final folder holding
`<id>.json`, rather than a base to which the checker appends a robot type.

Default calibration resolution reuses `HF_LEROBOT_CALIBRATION` and its
`HF_LEROBOT_HOME` / `HF_HOME` fallbacks from `lerobot.utils.constants`:

- Follower: `<calibration-base>/robots/so_follower/<id>.json`.
- Leader: `<calibration-base>/teleoperators/so_leader/<id>.json`.

The directory names are the implementations' class names, not their
`so101_follower` / `so101_leader` registry aliases. An omitted/null ID resolves
`None.json`, matching the runtime; the report recommends an explicit stable ID.
Missing calibration directories are never created by inspection.

The library interface is `lerobot.utils.setup_check.check_so101_setup(config)`;
it returns a `SetupReport`. `write_setup_report(report, Path(...))` writes both
formats. Output must be a new directory outside the inspected calibration folders;
existing directories, files and symlinks are refused. Choose a private report
location: the report contains selected setup paths and calibration evidence.

## Checks and evidence limits

| Category               | Inspected evidence                                                                                                      | Limit                                                                                                    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Configuration          | Supported types, required fields, joint units, camera shape/FPS settings, stream options and feature-name collisions    | Does not parse a full training/recording run or validate every device-specific register range            |
| Device paths           | File metadata, character-device type for serial paths, camera file/device paths, path aliases and duplicate assignments | Existence does not identify the arm/camera, prove communication or guarantee permission to open a device |
| Calibration files      | Six joint names/IDs, integer fields, drive modes, STS3215 offsets and encoder ranges                                    | Does not compare with live registers or establish that the file belongs to the physical arm              |
| Hardware communication | Not checked                                                                                                             | Normal SO connection methods mutate hardware                                                             |
| Camera frames          | Not checked                                                                                                             | Requested shape/FPS is distinct from actual shape, freshness and useful framing                          |
| Physical inspection    | Not checked                                                                                                             | Requires operator evidence about mounting, clearance, cables and framing                                 |

Both arms use joint IDs 1–6 in this order: `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. A calibration entry has
`id`, `drive_mode`, `homing_offset`, `range_min`, `range_max`. The supported
STS3215 contract uses drive modes 0/1, signed homing offsets in [-2047, 2047],
and `0 <= range_min < range_max <= 4095`. A full-turn wrist range of [0, 4095]
is valid. File consistency never proves that a calibration is physically correct.

OpenCV integer indices and RealSense selectors remain undiscovered. Equal
selectors and aliased local paths are detected, but different selectors or
backends can still refer to the same physical camera. On Linux, OpenCV's
default/V4L2 index is also compared with `/dev/video<index>` and its path aliases.
Character-device aliases are compared by device number. Camera paths assigned
to an arm serial port are flagged. Other cross-backend identity and the camera's
role cannot be inferred. Local OpenCV video
paths are checked for existence/type without decoding. Windows COM port names
are reported without probing them; case/prefix aliases are compared for duplicates.

Unknown nested configuration fields and unsupported camera/arm types produce
an incomplete report, preserving other findings. Missing calibration files are
warnings under an unchecked category, since calibration may not have been done
yet. Access errors are incomplete; malformed calibration JSON or inconsistent
stored joint data are verified faults. Duplicate JSON keys and non-finite numbers
are rejected rather than silently normalized. No files are repaired.

## Reports and exit codes

`setup.json` uses schema version `1.0`, with provenance, scope, inspection counts,
independent category states and findings. Each finding has a stable `check_id`,
classification, severity, artifact, expected/observed evidence, limits and next
action. Classification and severity are independent. `setup.md` presents the same
evidence for people. Whole run configurations and recordings are not exported.

- `0`: Supported file inspection completed without verified faults. Warnings and
  explicitly unverified hardware categories can remain.
- `1`: Supported file inspection completed and found verified faults.
- `2`: Inspection could not complete (input/access errors, unsupported scope or
  interruption), or the report could not be written. Available evidence is kept
  in reports when possible; a report-write failure prints the error.

`complete` applies only to the supported file-inspection scope. It never means
that all six verification categories passed. An interrupted inspection produces
an incomplete report; rerun it for complete file coverage.

Compatibility is grounded in the current `robots/so_follower`,
`teleoperators/so_leader`, base calibration loaders and OpenCV/RealSense config
implementations. Synthetic tests guard the joint/class-name source contract,
exercise the CLI with hardware imports and network blocked, and fingerprint input
files to detect writes. Live SO-101 and camera validation remains separate.
