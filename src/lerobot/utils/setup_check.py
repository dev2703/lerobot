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

"""Inspect SO-101 setup files without constructing or connecting hardware objects."""

import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from lerobot import __version__
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

Classification = Literal["verified_fault", "warning", "not_checked", "passed"]
Category = Literal[
    "configuration",
    "device_paths",
    "calibration_files",
    "hardware_communication",
    "camera_frames",
    "physical_inspection",
]

# SOFollower and SOLeader share this contract. A source-contract regression test
# guards these offline constants without importing optional hardware modules.
SO101_MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
CALIBRATION_NAMES = {"robot": "so_follower", "teleop": "so_leader"}
CATEGORIES: tuple[Category, ...] = (
    "configuration",
    "device_paths",
    "calibration_files",
    "hardware_communication",
    "camera_frames",
    "physical_inspection",
)
LIMITS = (
    "File and path inspection only. No port or camera was opened. Device identity, communication, "
    "motor registers, actual frame shape/freshness, physical mounting, workspace clearance, cable "
    "routing and camera framing remain unverified. This report does not certify robot safety."
)


@dataclass
class SetupFinding:
    """Evidence for one bounded setup check, with certainty separate from severity."""

    check_id: str
    category: Category
    classification: Classification
    severity: str
    artifact: str
    expected: str
    observed: Any
    next_action: str
    limits: str = "Configuration evidence does not establish hardware readiness."


@dataclass
class SetupReport:
    """Versioned report for the supported file-inspection scope."""

    schema_version: str = "1.0"
    complete: bool = True
    findings: list[SetupFinding] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=lambda: _identity())
    coverage: dict[str, int] = field(
        default_factory=lambda: {
            "arms": 0,
            "cameras": 0,
            "calibration_files": 0,
            "opened_devices": 0,
        }
    )

    def add(
        self,
        check_id: str,
        category: Category,
        classification: Classification,
        artifact: str,
        expected: str,
        observed: Any,
        next_action: str,
        *,
        severity: str | None = None,
        limits: str | None = None,
        incomplete: bool = False,
    ) -> None:
        """Append a finding and retain any earlier incompleteness."""
        finding = SetupFinding(
            check_id,
            category,
            classification,
            severity or {"verified_fault": "error", "warning": "warning"}.get(classification, "info"),
            artifact,
            expected,
            observed,
            next_action,
        )
        if limits is not None:
            finding.limits = limits
        self.findings.append(finding)
        self.complete = self.complete and not incomplete

    @property
    def exit_code(self) -> int:
        """Return 2 for incomplete inspection, 1 for faults, otherwise 0."""
        if not self.complete:
            return 2
        return int(any(f.classification == "verified_fault" for f in self.findings))

    def to_dict(self) -> dict[str, Any]:
        """Serialize evidence and independent verification categories."""
        result = asdict(self)
        result["scope"] = "so101_files_and_device_paths"
        result["limits"] = LIMITS
        result["exit_code"] = self.exit_code
        order: tuple[Classification, ...] = ("verified_fault", "not_checked", "warning", "passed")
        result["categories"] = {
            category: next(
                (
                    state
                    for state in order
                    if any(f.category == category and f.classification == state for f in self.findings)
                ),
                "not_checked",
            )
            for category in CATEGORIES
        }
        return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite JSON number: {value}")
    return result


def read_setup_json(path: Path) -> Any:
    """Read a regular JSON file, rejecting duplicate keys and non-finite numbers."""
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Expected a regular JSON file: {path}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )


def _identity() -> dict[str, Any]:
    source = Path(__file__).resolve().parents[1]
    result: dict[str, Any] = {"lerobot_version": __version__, "imported_source": str(source)}
    for name in ("lerobot", "rustypot", "opencv-python", "opencv-python-headless", "pyrealsense2"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    checkout = source.parents[1]
    result["checkout_revision"] = None
    git = shutil.which("git")
    if git is not None and (checkout / ".git").exists():
        try:
            revision = subprocess.run(
                [git, "-C", str(checkout), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            result["checkout_revision"] = revision.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def _integer(value: Any, minimum: int = 0, maximum: int | None = None) -> bool:
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def _positive(value: Any) -> bool:
    return (type(value) is int and value > 0) or (type(value) is float and math.isfinite(value) and value > 0)


def _value_check(report: SetupReport, artifact: str, valid: bool, expected: str, value: Any) -> None:
    report.add(
        "setup.config.value",
        "configuration",
        "passed" if valid else "verified_fault",
        artifact,
        expected,
        value,
        "Use a value matching the current configuration contract.",
    )


def _extra_fields(report: SetupReport, artifact: str, config: dict[str, Any], supported: set[str]) -> None:
    if extra := sorted(set(config) - supported):
        report.add(
            "setup.config.unsupported_fields",
            "configuration",
            "not_checked",
            artifact,
            "Fields covered by this checker's supported configuration contract",
            extra,
            "Review unrecognized fields against the installed implementation.",
            incomplete=True,
        )


def _path_identity(path: Path) -> tuple[Any, ...]:
    """Identify path aliases without opening devices, including hard links."""
    try:
        info = path.stat()
        if stat.S_ISCHR(info.st_mode) and info.st_rdev is not None:
            return ("device", info.st_rdev)
        return ("inode", info.st_dev, info.st_ino)
    except OSError:
        return ("path", os.path.realpath(path))


def _device_path(report: SetupReport, artifact: str, path: Path, *, serial: bool) -> None:
    try:
        info = path.stat()
    except FileNotFoundError:
        report.add(
            "setup.device.path",
            "device_paths",
            "verified_fault",
            artifact,
            "An existing device path",
            {"path": str(path), "exists": False},
            "Check the configured path and USB connection; do not use connect() as a probe.",
        )
        return
    except OSError as exc:
        report.add(
            "setup.device.path",
            "device_paths",
            "not_checked",
            artifact,
            "A path whose metadata can be inspected",
            str(exc),
            "Resolve the path or permission error and rerun file inspection.",
            incomplete=True,
        )
        return
    valid = stat.S_ISCHR(info.st_mode) or (not serial and stat.S_ISREG(info.st_mode))
    report.add(
        "setup.device.path",
        "device_paths",
        "passed" if valid else "verified_fault",
        artifact,
        "A character device" if serial else "A camera device or regular video file",
        {"path": str(path), "resolved": os.path.realpath(path), "mode": stat.S_IFMT(info.st_mode)},
        "Verify device identity and communication separately under an inspected access procedure.",
        limits="Path existence/type does not identify the attached hardware or establish permission to open it.",
    )


def _calibration(report: SetupReport, role: str, config: dict[str, Any]) -> None:
    identifier = config.get("id")
    if identifier is None:
        report.add(
            "setup.calibration.id",
            "configuration",
            "warning",
            f"{role}.id",
            "An explicit stable calibration ID",
            None,
            "Prefer a stable ID and reuse it for calibration, recording and inference."
            " This configuration resolves None.json, just as the runtime does.",
        )
    elif (
        not isinstance(identifier, str)
        or not identifier
        or identifier in (".", "..")
        or any(char in identifier for char in ("/", "\\", "\0"))
    ):
        _value_check(
            report, f"{role}.id", False, "A nonempty calibration filename without path separators", identifier
        )
        return
    directory = config.get("calibration_dir")
    if directory is not None and (not isinstance(directory, str) or not directory or "\0" in directory):
        _value_check(report, f"{role}.calibration_dir", False, "A directory path or null", directory)
        return
    # Explicit calibration_dir is already the final folder. Runtime defaults use
    # class.name (so_follower/so_leader), not the registry alias so101_*.
    base = (
        Path(directory)
        if directory is not None
        else (
            HF_LEROBOT_CALIBRATION
            / ("robots" if role == "robot" else "teleoperators")
            / CALIBRATION_NAMES[role]
        )
    )
    path = base / f"{identifier}.json"
    artifact = str(path.absolute())
    try:
        data = read_setup_json(path)
    except FileNotFoundError:
        report.add(
            "setup.calibration.file",
            "calibration_files",
            "not_checked",
            artifact,
            "An existing calibration file",
            "Missing local file",
            "Locate the intended calibration file/ID."
            " Calibration itself requires a separately scoped hardware workflow.",
            severity="warning",
        )
        return
    except OSError as exc:
        report.add(
            "setup.calibration.file",
            "calibration_files",
            "not_checked",
            artifact,
            "A readable regular calibration file",
            str(exc),
            "Resolve file access and rerun inspection.",
            incomplete=True,
        )
        return
    except (ValueError, UnicodeError) as exc:
        report.add(
            "setup.calibration.file",
            "calibration_files",
            "verified_fault",
            artifact,
            "An unambiguous JSON calibration object",
            str(exc),
            "Review the file without rewriting it automatically.",
        )
        return
    report.coverage["calibration_files"] += 1
    valid = isinstance(data, dict) and set(data) == set(SO101_MOTOR_IDS)
    if valid:
        for motor, motor_id in SO101_MOTOR_IDS.items():
            entry = data[motor]
            if not isinstance(entry, dict) or set(entry) != {
                "id",
                "drive_mode",
                "homing_offset",
                "range_min",
                "range_max",
            }:
                valid = False
                break
            valid = (
                type(entry["id"]) is int
                and entry["id"] == motor_id
                and _integer(entry["drive_mode"], 0, 1)
                and _integer(entry["homing_offset"], -2047, 2047)
                and _integer(entry["range_min"], 0, 4095)
                and _integer(entry["range_max"], 0, 4095)
                and entry["range_min"] < entry["range_max"]
            )
            if not valid:
                break
    observed = (
        {
            "joint_names": sorted(data),
            "joints": {
                name: {
                    key: entry[key]
                    for key in ("id", "drive_mode", "homing_offset", "range_min", "range_max")
                    if key in entry
                }
                if isinstance(entry, dict)
                else type(entry).__name__
                for name, entry in data.items()
                if name in SO101_MOTOR_IDS
            },
        }
        if isinstance(data, dict)
        else {"json_type": type(data).__name__}
    )
    report.add(
        "setup.calibration.contract",
        "calibration_files",
        "passed" if valid else "verified_fault",
        artifact,
        "Six STS3215 joints with IDs 1–6, integer calibration fields, drive mode 0/1, "
        "offset in [-2047, 2047] and 0 <= range_min < range_max <= 4095",
        observed,
        "Review mismatched joint names, IDs or encoder ranges; compare motor registers only"
        " through a separately authorized procedure.",
        limits="File consistency does not establish correspondence with the current physical arm or its motor registers.",
    )


def _cameras(report: SetupReport, cameras: Any, arm_ports: dict[tuple[Any, ...], str]) -> None:
    if not isinstance(cameras, dict):
        _value_check(
            report, "robot.cameras", False, "An object mapping feature names to camera configs", cameras
        )
        return
    if not cameras:
        report.add(
            "setup.camera.none",
            "configuration",
            "warning",
            "robot.cameras",
            "Cameras when the intended task requires vision",
            "No cameras configured",
            "Confirm whether this is intentional for the task.",
        )
    sources: dict[tuple[Any, ...], str] = {}
    features = {f"{motor}.pos" for motor in SO101_MOTOR_IDS}
    for key, camera in cameras.items():
        artifact = f"robot.cameras.{key}"
        report.coverage["cameras"] += 1
        if not isinstance(key, str) or not key or not isinstance(camera, dict):
            _value_check(report, artifact, False, "A nonempty feature name and camera object", camera)
            continue
        camera_type = camera.get("type")
        if camera_type not in ("opencv", "intelrealsense"):
            report.add(
                "setup.camera.type",
                "configuration",
                "not_checked",
                artifact,
                "opencv or intelrealsense",
                camera_type,
                "Inspect this camera plugin's contract separately.",
                incomplete=True,
            )
            continue
        common = {"type", "width", "height", "fps", "color_mode", "rotation", "warmup_s"}
        specific = (
            {"index_or_path", "fourcc", "backend"}
            if camera_type == "opencv"
            else {
                "serial_number_or_name",
                "use_rgb",
                "use_depth",
                "exposure",
                "gain",
                "white_balance",
            }
        )
        _extra_fields(report, artifact, camera, common | specific)
        for name in ("width", "height", "fps"):
            _value_check(
                report,
                f"{artifact}.{name}",
                _integer(camera.get(name), 1),
                "A positive integer required by RobotConfig",
                camera.get(name),
            )
        for name, default, allowed in (
            ("color_mode", "rgb", ("rgb", "bgr")),
            ("rotation", 0, (0, 90, 180, -90)),
        ):
            value = camera.get(name, default)
            valid = type(value) is type(default) and value in allowed
            _value_check(report, f"{artifact}.{name}", valid, f"One of {allowed}", value)
        if "warmup_s" in camera:
            _value_check(
                report,
                f"{artifact}.warmup_s",
                _integer(camera["warmup_s"]),
                "A nonnegative integer",
                camera["warmup_s"],
            )
        source: tuple[Any, ...] | None = None
        use_rgb, use_depth = True, False
        if camera_type == "opencv":
            value = camera.get("index_or_path")
            valid = _integer(value) or (isinstance(value, str) and bool(value) and "\0" not in value)
            _value_check(
                report,
                f"{artifact}.index_or_path",
                valid,
                "A nonnegative integer index or nonempty local path",
                value,
            )
            if _integer(value):
                backend = camera.get("backend", 0)
                backend_key = backend if type(backend) is int else None
                source = (
                    ("opencv", *_path_identity(Path(f"/dev/video{value}")))
                    if sys.platform.startswith("linux") and backend_key in (0, 200)
                    else ("opencv-index", value, backend_key)
                )
                report.add(
                    "setup.camera.discovery",
                    "device_paths",
                    "not_checked",
                    artifact,
                    "Camera identity evidence",
                    {"index": value},
                    "Identify this index separately; enumeration/capture is outside this file-only check.",
                )
            elif valid and isinstance(value, str):
                path = Path(value)
                source = ("opencv", *_path_identity(path))
                _device_path(report, artifact, path, serial=False)
                if source[1:] in arm_ports:
                    report.add(
                        "setup.device.assignment",
                        "configuration",
                        "verified_fault",
                        artifact,
                        "Camera paths distinct from arm serial ports",
                        {"also_assigned_to": arm_ports[source[1:]]},
                        "Correct the camera/arm path assignment before device access.",
                    )
            if "fourcc" in camera and camera["fourcc"] is not None:
                value = camera["fourcc"]
                _value_check(
                    report,
                    f"{artifact}.fourcc",
                    isinstance(value, str) and len(value) == 4,
                    "A four-character string or null",
                    value,
                )
            value = camera.get("backend", 0)
            _value_check(
                report,
                f"{artifact}.backend",
                type(value) is int and value in (0, 200, 700, 800, 1000, 1200, 1400),
                "A supported Cv2Backends value",
                value,
            )
        else:
            value = camera.get("serial_number_or_name")
            valid = isinstance(value, str) and bool(value.strip())
            _value_check(
                report, f"{artifact}.serial_number_or_name", valid, "A nonempty device selector", value
            )
            if valid:
                source = ("intelrealsense", value)
            use_rgb, use_depth = camera.get("use_rgb", True), camera.get("use_depth", False)
            for name, value in (("use_rgb", use_rgb), ("use_depth", use_depth)):
                _value_check(report, f"{artifact}.{name}", type(value) is bool, "A boolean", value)
            _value_check(
                report,
                artifact,
                bool(use_rgb or use_depth),
                "At least one enabled stream",
                {"use_rgb": use_rgb, "use_depth": use_depth},
            )
            for name in ("exposure", "gain", "white_balance"):
                if camera.get(name) is not None:
                    _value_check(
                        report,
                        f"{artifact}.{name}",
                        type(camera[name]) is int and use_rgb is True,
                        "An integer with use_rgb=true; device-specific ranges are unchecked",
                        camera[name],
                    )
            report.add(
                "setup.camera.discovery",
                "device_paths",
                "not_checked",
                artifact,
                "Camera identity evidence",
                {"selector": camera.get("serial_number_or_name")},
                "Verify the selector separately; no RealSense SDK is imported by this check.",
            )
        if source is not None:
            if source in sources:
                report.add(
                    "setup.camera.duplicate",
                    "configuration",
                    "verified_fault",
                    artifact,
                    "A separate source for each camera",
                    {"also_assigned_to": sources[source]},
                    "Review duplicate camera assignments before device access.",
                )
            sources[source] = key
        emitted = ([key] if use_rgb else []) + ([f"{key}_depth"] if use_depth else [])
        for feature in emitted:
            _value_check(
                report,
                artifact,
                feature not in features,
                "An observation feature name distinct from other cameras and motor features",
                feature,
            )
            features.add(feature)


def check_so101_setup(config: Any) -> SetupReport:
    """Inspect native JSON robot/teleop configuration fields without hardware access.

    Args:
        config: A JSON object containing a ``robot`` SO-101 follower configuration
            and, optionally, a ``teleop`` SO-101 leader configuration. Other
            top-level run settings are outside this inspection's scope.

    Returns:
        A versioned report of configuration, path and calibration-file evidence.
            Runtime communication and human inspection remain explicitly unchecked.
    """
    report = SetupReport()
    try:
        return _inspect_so101_setup(config, report)
    except KeyboardInterrupt:
        report.add(
            "setup.interrupted",
            "configuration",
            "not_checked",
            "setup",
            "Completed file inspection",
            "Interrupted; earlier findings retained",
            "Rerun to inspect all configured files.",
            incomplete=True,
        )
        return report


def _inspect_so101_setup(config: Any, report: SetupReport) -> SetupReport:
    unchecked: tuple[tuple[str, Category, str, str], ...] = (
        (
            "setup.communication",
            "hardware_communication",
            "Read-only hardware communication evidence",
            "Inspect initialization, read and cleanup effects before separately authorizing device access.",
        ),
        (
            "setup.frames",
            "camera_frames",
            "Actual frame shape and freshness",
            "Separately authorize camera capture; inspect actual frames and task-specific framing.",
        ),
        (
            "setup.physical",
            "physical_inspection",
            "Human inspection of mounting, clearance and cabling",
            "Have the operator inspect mounting, workspace clearance, cable routing and camera framing.",
        ),
    )
    for check_id, category, expected, next_action in unchecked:
        report.add(
            check_id,
            category,
            "not_checked",
            "setup",
            expected,
            "No live inspection performed",
            next_action,
            limits=LIMITS,
        )
    if not isinstance(config, dict) or "robot" not in config:
        report.add(
            "setup.config.input",
            "configuration",
            "not_checked",
            "config",
            "A JSON object with a robot configuration",
            "Missing robot configuration",
            "Supply a saved run configuration or an object with robot and optional teleop fields.",
            incomplete=True,
        )
        return report
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        report.add(
            "setup.config.input",
            "configuration",
            "not_checked",
            "config",
            "JSON-compatible values with finite numbers",
            str(exc),
            "Supply a valid JSON configuration.",
            incomplete=True,
        )
        return report
    ports: dict[tuple[Any, ...], str] = {}
    arms: dict[str, dict[str, Any]] = {}
    for role, expected_type, alias in (
        ("robot", "so101_follower", "so100_follower"),
        ("teleop", "so101_leader", "so100_leader"),
    ):
        if role == "teleop" and config.get(role) is None:
            continue
        arm = config[role]
        if not isinstance(arm, dict) or arm.get("type") not in (expected_type, alias):
            report.add(
                "setup.arm.type",
                "configuration",
                "not_checked",
                role,
                expected_type,
                arm.get("type") if isinstance(arm, dict) else type(arm).__name__,
                "Use the SO-101 configuration or inspect the other hardware workflow separately.",
                incomplete=True,
            )
            continue
        arms[role] = arm
        report.coverage["arms"] += 1
        common = {"type", "port", "id", "calibration_dir", "use_degrees", "num_read_retries"}
        follower_fields = {
            "cameras",
            "max_relative_target",
            "disable_torque_on_disconnect",
            "position_p_coefficient",
            "position_i_coefficient",
            "position_d_coefficient",
        }
        _extra_fields(report, role, arm, common | (follower_fields if role == "robot" else set()))
        _value_check(
            report, f"{role}.type", True, f"{expected_type} or its shared {alias} alias", arm["type"]
        )
        if arm["type"] == alias:
            report.add(
                "setup.arm.alias",
                "configuration",
                "warning",
                f"{role}.type",
                "An SO-101 configuration; the common SO implementation also serializes SO-100 aliases",
                alias,
                "Confirm the intended physical arm separately; the shared configuration alias cannot identify it.",
            )
        port = arm.get("port")
        if not isinstance(port, str) or not port or "\0" in port:
            _value_check(report, f"{role}.port", False, "A nonempty serial-device path", port)
        else:
            windows_port = re.fullmatch(r"(?:\\\\\.\\)?COM\d+", port, re.IGNORECASE)
            identity = (
                ("com", port.upper().removeprefix("\\\\.\\")) if windows_port else _path_identity(Path(port))
            )
            if identity in ports:
                report.add(
                    "setup.arm.duplicate_port",
                    "configuration",
                    "verified_fault",
                    f"{role}.port",
                    "Distinct follower and leader ports",
                    {"path": port, "also_assigned_to": ports[identity]},
                    "Correct the port assignment; file discovery does not identify which arm is attached.",
                )
            ports[identity] = role
            if windows_port:
                report.add(
                    "setup.device.path",
                    "device_paths",
                    "not_checked",
                    f"{role}.port",
                    "OS evidence identifying the serial device",
                    port,
                    "COM-port discovery is unavailable in this file-only checker; inspect OS evidence separately.",
                )
            else:
                _device_path(report, f"{role}.port", Path(port), serial=True)
        value = arm.get("use_degrees", True)
        _value_check(report, f"{role}.use_degrees", type(value) is bool, "A boolean", value)
        if "num_read_retries" in arm:
            _value_check(
                report,
                f"{role}.num_read_retries",
                _integer(arm["num_read_retries"]),
                "A nonnegative integer",
                arm["num_read_retries"],
            )
        if role == "robot":
            for name in ("position_p_coefficient", "position_i_coefficient", "position_d_coefficient"):
                if name in arm:
                    _value_check(
                        report,
                        f"{role}.{name}",
                        _integer(arm[name]),
                        "A nonnegative integer; device register limits are unchecked",
                        arm[name],
                    )
            if "disable_torque_on_disconnect" in arm:
                _value_check(
                    report,
                    f"{role}.disable_torque_on_disconnect",
                    type(arm["disable_torque_on_disconnect"]) is bool,
                    "A boolean",
                    arm["disable_torque_on_disconnect"],
                )
            cap = arm.get("max_relative_target")
            valid_cap = (
                cap is None
                or _positive(cap)
                or (
                    isinstance(cap, dict)
                    and set(cap) == set(SO101_MOTOR_IDS)
                    and all(_positive(v) for v in cap.values())
                )
            )
            _value_check(
                report,
                f"{role}.max_relative_target",
                valid_cap,
                "null, a finite positive scalar, or a positive cap for every joint",
                cap,
            )
        _calibration(report, role, arm)
    if "robot" in arms:
        _cameras(report, arms["robot"].get("cameras", {}), ports)
    if len(arms) == 2:
        follower, leader = (arms[role].get("use_degrees", True) for role in ("robot", "teleop"))
        report.add(
            "setup.arm.units",
            "configuration",
            "passed" if follower == leader else "verified_fault",
            "robot.use_degrees / teleop.use_degrees",
            "Matching leader/follower position units",
            {"robot": follower, "teleop": leader},
            "Use matching units for direct leader/follower teleoperation.",
            limits="Custom processors may intentionally convert units; their behavior is outside this check.",
        )
    return report


def _markdown_value(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "&#96;")
    )


def write_setup_report(report: SetupReport, output_dir: Path) -> None:
    """Write JSON and Markdown to a new directory without overwriting existing files.

    Args:
        report: Completed or partial file-inspection evidence.
        output_dir: A new report directory. Existing paths are refused.
    """
    resolved_output = output_dir.resolve()
    for finding in report.findings:
        if finding.category == "calibration_files" and resolved_output.is_relative_to(
            Path(finding.artifact).resolve().parent
        ):
            raise ValueError("Report output must be outside calibration directories")
    data = report.to_dict()
    encoded = json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    lines = [
        "# SO-101 setup inspection",
        "",
        f"File inspection complete: {report.complete}. Exit code: {report.exit_code}.",
        "",
        LIMITS,
        "",
        "## Verification categories",
        "",
    ]
    lines.extend(f"- {category}: **{state}**" for category, state in data["categories"].items())
    lines.extend(["", "## Provenance", "", f"`{_markdown_value(report.provenance)}`", "", "## Findings", ""])
    for finding in report.findings:
        lines.extend(
            [
                f"### {finding.check_id} — {finding.classification} ({finding.severity})",
                "",
                f"Artifact: `{_markdown_value(finding.artifact)}`",
                "",
                f"Expected: {finding.expected}",
                "",
                f"Observed: `{_markdown_value(finding.observed)}`",
                "",
                f"Limit: {finding.limits}",
                "",
                f"Next action: {finding.next_action}",
                "",
            ]
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "setup.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    with (output_dir / "setup.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
