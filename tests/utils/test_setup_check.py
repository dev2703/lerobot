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

import ast
import copy
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from lerobot.scripts.lerobot_check_setup import main
from lerobot.utils import setup_check as checker

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"


def calibration() -> dict[str, dict[str, int]]:
    return {
        name: {"id": index, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for name, index in checker.SO101_MOTOR_IDS.items()
    }


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Provide real JSON files and stat-only fake serial devices, without hardware extras."""
    config = {}
    device_files = set()
    for role, arm_type in (("robot", "so101_follower"), ("teleop", "so101_leader")):
        directory = tmp_path / role
        directory.mkdir()
        (directory / "arm.json").write_text(json.dumps(calibration()))
        port = tmp_path / f"{role}.device"
        port.write_bytes(b"synthetic device - must never be opened by the checker")
        device_files.add(port)
        config[role] = {"type": arm_type, "port": str(port), "id": "arm", "calibration_dir": str(directory)}
    config["robot"]["cameras"] = {
        "front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    }
    original_stat = Path.stat

    def device_stat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        info = original_stat(path, *args, **kwargs)
        if path in device_files:
            values = list(info)
            values[0] = stat.S_IFCHR | stat.S_IMODE(info.st_mode)
            return os.stat_result(values)
        return info

    monkeypatch.setattr(Path, "stat", device_stat)
    return config


def faults(report: checker.SetupReport, check_id: str | None = None) -> list[checker.SetupFinding]:
    return [
        f
        for f in report.findings
        if f.classification == "verified_fault" and (check_id is None or f.check_id == check_id)
    ]


def test_valid_files_keep_hardware_and_camera_claims_unverified(setup: dict[str, Any]) -> None:
    report = checker.check_so101_setup(setup)
    assert report.complete and report.exit_code == 0 and not faults(report)
    data = report.to_dict()
    assert data["coverage"] == {"arms": 2, "cameras": 1, "calibration_files": 2, "opened_devices": 0}
    assert data["categories"] == {
        "configuration": "passed",
        "device_paths": "not_checked",
        "calibration_files": "passed",
        "hardware_communication": "not_checked",
        "camera_frames": "not_checked",
        "physical_inspection": "not_checked",
    }
    assert report.provenance["imported_source"] == str(SOURCE / "lerobot")
    assert report.provenance["checkout_revision"]


def test_inference_without_leader_or_cameras_is_supported(setup: dict[str, Any]) -> None:
    setup["teleop"] = None
    setup["robot"]["cameras"] = {}
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 0
    assert report.coverage["arms"] == 1
    assert any(f.check_id == "setup.camera.none" and f.classification == "warning" for f in report.findings)


@pytest.mark.parametrize("config", [None, [], {}, {"robot": None}, {"robot": {"type": "koch_follower"}}])
def test_unsupported_or_missing_input_is_incomplete(config: Any) -> None:
    report = checker.check_so101_setup(config)
    assert report.exit_code == 2 and not report.complete


@pytest.mark.parametrize(
    "name,value",
    [
        ("port", None),
        ("port", ""),
        ("port", 7),
        ("port", "a\0b"),
        ("id", "../escape"),
        ("id", "a\\b"),
        ("id", ""),
        ("id", True),
        ("calibration_dir", False),
        ("calibration_dir", ""),
        ("use_degrees", "true"),
        ("num_read_retries", True),
        ("num_read_retries", -1),
        ("disable_torque_on_disconnect", 1),
        ("position_p_coefficient", -1),
        ("max_relative_target", -1),
        ("max_relative_target", "10"),
        ("max_relative_target", {"gripper": 1}),
        ("cameras", None),
    ],
)
def test_invalid_arm_values_are_faults(setup: dict[str, Any], name: str, value: Any) -> None:
    setup["robot"][name] = value
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 1 and faults(report)


@pytest.mark.parametrize("cap", [1, 0.1, None, dict.fromkeys(checker.SO101_MOTOR_IDS, 2.0), 10**400])
def test_valid_scalar_and_per_joint_caps(setup: dict[str, Any], cap: Any) -> None:
    setup["robot"]["max_relative_target"] = cap
    assert checker.check_so101_setup(setup).exit_code == 0


def test_unknown_nested_fields_are_explicitly_unsupported(setup: dict[str, Any]) -> None:
    setup["robot"]["motor_mapping"] = {"unexpected": 42}
    setup["robot"]["cameras"]["front"]["unknown_option"] = True
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 2
    assert len([f for f in report.findings if f.check_id == "setup.config.unsupported_fields"]) == 2


def test_units_mismatch_and_duplicate_ports(setup: dict[str, Any]) -> None:
    setup["teleop"]["port"] = setup["robot"]["port"]
    setup["teleop"]["use_degrees"] = False
    report = checker.check_so101_setup(setup)
    assert faults(report, "setup.arm.duplicate_port") and faults(report, "setup.arm.units")


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_duplicate_port_aliases(setup: dict[str, Any], tmp_path: Path, alias: str) -> None:
    original = Path(setup["robot"]["port"])
    other = tmp_path / "alias.device"
    if alias == "symlink":
        other.symlink_to(original)
    else:
        os.link(original, other)
    setup["teleop"]["port"] = str(other)
    assert faults(checker.check_so101_setup(setup), "setup.arm.duplicate_port")


@pytest.mark.parametrize("port", ["COM3", "com3", "\\\\.\\COM3"])
def test_com_ports_are_not_probed_and_duplicates_are_case_insensitive(
    setup: dict[str, Any], port: str
) -> None:
    setup["robot"]["port"] = port
    setup["teleop"]["port"] = "COM3"
    report = checker.check_so101_setup(setup)
    assert faults(report, "setup.arm.duplicate_port")
    assert not faults(report, "setup.device.path")


@pytest.mark.parametrize("kind", ["missing", "broken-symlink", "regular-file", "directory"])
def test_invalid_serial_paths(setup: dict[str, Any], tmp_path: Path, kind: str) -> None:
    path = tmp_path / "other.device"
    if kind == "regular-file":
        path.write_text("not a serial device")
    elif kind == "directory":
        path.mkdir()
    elif kind == "broken-symlink":
        path.symlink_to(tmp_path / "missing")
    setup["robot"]["port"] = str(path)
    assert faults(checker.check_so101_setup(setup), "setup.device.path")


def test_permission_error_preserves_other_findings(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.stat
    blocked = Path(setup["robot"]["port"])

    def denied(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == blocked:
            raise PermissionError("synthetic denied metadata")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 2
    assert report.coverage["calibration_files"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 2),
        ("id", True),
        ("drive_mode", 2),
        ("drive_mode", False),
        ("homing_offset", 2048),
        ("homing_offset", -2048),
        ("homing_offset", 1.0),
        ("range_min", -1),
        ("range_min", 4095),
        ("range_max", 4096),
        ("range_max", 0),
    ],
)
def test_calibration_contract_faults(setup: dict[str, Any], field: str, value: Any) -> None:
    path = Path(setup["robot"]["calibration_dir"]) / "arm.json"
    data = calibration()
    data["shoulder_pan"][field] = value
    path.write_text(json.dumps(data))
    report = checker.check_so101_setup(setup)
    assert faults(report, "setup.calibration.contract")
    assert report.to_dict()["categories"]["calibration_files"] == "verified_fault"


@pytest.mark.parametrize("data", [None, [], {}, {"other_joint": {"id": 1}}])
def test_calibration_wrong_structure(setup: dict[str, Any], data: Any) -> None:
    path = Path(setup["robot"]["calibration_dir"]) / "arm.json"
    path.write_text(json.dumps(data))
    assert faults(checker.check_so101_setup(setup), "setup.calibration.contract")


def test_missing_calibration_is_unchecked_without_creating_directories(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    directory = tmp_path / "does-not-exist"
    setup["robot"]["calibration_dir"] = str(directory)
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 0
    assert not directory.exists()
    assert report.to_dict()["categories"]["calibration_files"] == "not_checked"


def test_default_calibration_paths_use_class_names_and_none_id(
    setup: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checker, "HF_LEROBOT_CALIBRATION", tmp_path / "calibration")
    for role, group, name in (("robot", "robots", "so_follower"), ("teleop", "teleoperators", "so_leader")):
        setup[role].pop("calibration_dir")
        setup[role].pop("id")
        directory = tmp_path / "calibration" / group / name
        directory.mkdir(parents=True)
        (directory / "None.json").write_text(json.dumps(calibration()))
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 0 and report.coverage["calibration_files"] == 2
    assert len([f for f in report.findings if f.check_id == "setup.calibration.id"]) == 2


@pytest.mark.parametrize(
    "name,value",
    [
        ("width", 0),
        ("height", True),
        ("fps", None),
        ("fps", 30.5),
        ("index_or_path", -1),
        ("index_or_path", True),
        ("index_or_path", ""),
        ("rotation", 270),
        ("rotation", False),
        ("color_mode", "RGB"),
        ("fourcc", "X"),
        ("backend", 9999),
        ("backend", []),
        ("backend", {}),
        ("warmup_s", -1),
    ],
)
def test_invalid_camera_values(setup: dict[str, Any], name: str, value: Any) -> None:
    setup["robot"]["cameras"]["front"][name] = value
    assert checker.check_so101_setup(setup).exit_code == 1


def test_duplicate_cameras_and_observation_feature_collisions(setup: dict[str, Any]) -> None:
    camera = setup["robot"]["cameras"]["front"]
    setup["robot"]["cameras"]["shoulder_pan.pos"] = copy.deepcopy(camera)
    report = checker.check_so101_setup(setup)
    assert faults(report, "setup.camera.duplicate")
    assert any(f.observed == "shoulder_pan.pos" for f in faults(report))


def test_camera_path_alias_and_regular_video_source(setup: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"discovery only: this must not be decoded")
    alias = tmp_path / "video-alias.mp4"
    alias.symlink_to(path)
    camera = setup["robot"]["cameras"]["front"]
    camera["index_or_path"] = str(path)
    setup["robot"]["cameras"]["second"] = {**camera, "index_or_path": str(alias)}
    report = checker.check_so101_setup(setup)
    assert not faults(report, "setup.device.path")
    assert faults(report, "setup.camera.duplicate")


def realsense(selector: str = "123") -> dict[str, Any]:
    return {
        "type": "intelrealsense",
        "serial_number_or_name": selector,
        "width": 640,
        "height": 480,
        "fps": 30,
    }


def test_realsense_without_sdk_including_depth_feature_collisions(setup: dict[str, Any]) -> None:
    setup["robot"]["cameras"] = {"front": {**realsense(), "use_depth": True}}
    assert checker.check_so101_setup(setup).exit_code == 0
    setup["robot"]["cameras"]["front_depth"] = realsense("other")
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 1
    assert any(f.observed == "front_depth" for f in faults(report))


@pytest.mark.parametrize(
    "changes",
    [
        {"use_rgb": False, "use_depth": False},
        {"use_depth": "true"},
        {"serial_number_or_name": " "},
        {"use_rgb": False, "use_depth": True, "gain": 2},
        {"exposure": 1.5},
    ],
)
def test_realsense_invalid_config_without_device_access(
    setup: dict[str, Any], changes: dict[str, Any]
) -> None:
    setup["robot"]["cameras"] = {"front": {**realsense(), **changes}}
    assert checker.check_so101_setup(setup).exit_code == 1


def test_unsupported_camera_preserves_other_inspection(setup: dict[str, Any]) -> None:
    setup["robot"]["cameras"]["front"]["type"] = "third_party_camera"
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 2 and report.coverage["calibration_files"] == 2


@pytest.mark.parametrize(
    "content",
    [
        '{"robot": {}, "robot": {}}',
        '{"robot": {"fps": NaN}}',
        '{"robot": {"fps": Infinity}}',
        '{"robot": {"fps": 1e999}}',
        '{"robot":',
    ],
)
def test_ambiguous_or_invalid_json_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.json"
    path.write_text(content)
    output = tmp_path / "report"
    assert main(["--config", str(path), "--output-dir", str(output)]) == 2
    data = json.loads((output / "setup.json").read_text())
    assert not data["complete"] and data["categories"]["hardware_communication"] == "not_checked"


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_api_input_is_incomplete(setup: dict[str, Any], number: float) -> None:
    setup["robot"]["max_relative_target"] = number
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 2
    json.dumps(report.to_dict(), allow_nan=False)


def test_cli_roundtrip_no_input_writes_or_network(
    setup: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup["dataset"] = {"repo_id": "private/task", "token": "must-not-be-copied"}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(setup))
    inputs = [p for p in tmp_path.rglob("*") if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network access attempted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    output = tmp_path / "report"
    assert main(["--config", str(path), "--output-dir", str(output)]) == 0
    report = json.loads((output / "setup.json").read_text())
    assert report["exit_code"] == 0
    assert report["provenance"]["config_path"] == str(path)
    assert "must-not-be-copied" not in (output / "setup.json").read_text()
    assert "hardware_communication: **not_checked**" in (output / "setup.md").read_text()
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}


def test_cli_returns_fault_exit_code(setup: dict[str, Any], tmp_path: Path) -> None:
    setup["robot"]["port"] = str(tmp_path / "missing")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(setup))
    assert main(["--config", str(path), "--output-dir", str(tmp_path / "report")]) == 1


def test_existing_output_and_symlink_are_never_overwritten(setup: dict[str, Any], tmp_path: Path) -> None:
    report = checker.check_so101_setup(setup)
    output = tmp_path / "report"
    checker.write_setup_report(report, output)
    before = (output / "setup.json").read_bytes()
    with pytest.raises(FileExistsError):
        checker.write_setup_report(report, output)
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(FileExistsError):
        checker.write_setup_report(report, alias)
    assert (output / "setup.json").read_bytes() == before


def test_output_inside_calibration_directory_is_refused(setup: dict[str, Any], tmp_path: Path) -> None:
    report = checker.check_so101_setup(setup)
    directory = Path(setup["robot"]["calibration_dir"])
    alias = tmp_path / "calibration-alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="outside calibration"):
        checker.write_setup_report(report, alias / "reports")
    assert not (directory / "reports").exists()


def test_interrupted_cli_reports_incompleteness(
    setup: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(setup))

    def interrupt(*args: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("lerobot.scripts.lerobot_check_setup.check_so101_setup", interrupt)
    output = tmp_path / "report"
    assert main(["--config", str(path), "--output-dir", str(output)]) == 2
    assert json.loads((output / "setup.json").read_text())["complete"] is False


def test_source_contract_matches_actual_so_implementations() -> None:
    """Fail when the offline model assumptions drift from either arm implementation."""
    for role, relative in (
        ("robot", "robots/so_follower/so_follower.py"),
        ("teleop", "teleoperators/so_leader/so_leader.py"),
    ):
        tree = ast.parse((SOURCE / "lerobot" / relative).read_text())
        names = [
            n.value.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "name" for t in n.targets)
            and isinstance(n.value, ast.Constant)
        ]
        assert checker.CALIBRATION_NAMES[role] in names
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "motors" and isinstance(node.value, ast.Dict):
                mapping = {
                    ast.literal_eval(key): ast.literal_eval(value.args[0])
                    for key, value in zip(node.value.keys, node.value.values, strict=True)
                }
                assert mapping == checker.SO101_MOTOR_IDS
                assert all(ast.literal_eval(value.args[1]) == "sts3215" for value in node.value.values)
                break
        else:
            pytest.fail(f"Motor mapping not found in {relative}")


def test_cli_imports_no_hardware_even_in_fresh_process(tmp_path: Path) -> None:
    """Guard imports, device opens and network use before importing the checker."""
    script = """
import builtins, io, pathlib, socket, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    roots = ("lerobot.robots", "lerobot.teleoperators", "lerobot.motors", "lerobot.cameras",
             "rustypot", "cv2", "pyrealsense2", "serial")
    if any(name == root or name.startswith(root + ".") for root in roots):
        raise AssertionError("Hardware import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def forbidden(*args, **kwargs):
    raise AssertionError("Network access attempted")
socket.create_connection = forbidden
socket.socket.connect = forbidden
original_open = io.open
def guarded_open(path, *args, **kwargs):
    if str(path).startswith("/dev/"):
        raise AssertionError("Device opened: " + str(path))
    return original_open(path, *args, **kwargs)
io.open = guarded_open
builtins.open = guarded_open
from lerobot.scripts.lerobot_check_setup import main
raise SystemExit(main(sys.argv[1:]))
"""
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "robot": {
                    "type": "so101_follower",
                    "port": "COM3",
                    "calibration_dir": str(tmp_path / "missing-calibration"),
                }
            }
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script, "--config", str(config), "--output-dir", str(tmp_path / "report")],
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_native_draccus_camera_enum_json_roundtrip(setup: dict[str, Any]) -> None:
    import draccus

    from lerobot.cameras.configs import ColorMode, Cv2Backends, Cv2Rotation

    camera = setup["robot"]["cameras"]["front"]
    camera.update(color_mode=ColorMode.BGR, rotation=Cv2Rotation.ROTATE_90, backend=Cv2Backends.ANY)
    saved = json.loads(json.dumps(draccus.encode(setup)))
    assert saved["robot"]["cameras"]["front"]["rotation"] == 90
    assert checker.check_so101_setup(saved).exit_code == 0


def test_camera_path_cannot_also_be_a_leader_port(setup: dict[str, Any]) -> None:
    setup["robot"]["cameras"]["front"]["index_or_path"] = setup["teleop"]["port"]
    assert faults(checker.check_so101_setup(setup), "setup.device.assignment")


def test_linux_index_and_video_path_alias_are_duplicates(
    setup: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "video-device"
    path.write_bytes(b"stat only")
    original_stat = Path.stat

    def video_stat(candidate: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        return original_stat(path if candidate == Path("/dev/video0") else candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", video_stat)
    monkeypatch.setattr(sys, "platform", "linux")
    camera = setup["robot"]["cameras"]["front"]
    setup["robot"]["cameras"]["second"] = {**camera, "index_or_path": str(path)}
    assert faults(checker.check_so101_setup(setup), "setup.camera.duplicate")


def test_character_device_aliases_use_device_number(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    def device_stat(path: Path) -> Any:
        return SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=123, st_dev=1, st_ino=hash(path))

    monkeypatch.setattr(Path, "stat", device_stat)
    assert checker._path_identity(Path("/device-a")) == checker._path_identity(Path("/device-b"))


def test_missing_input_cannot_be_created_as_report_output(tmp_path: Path) -> None:
    output = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(output / "setup.json"), "--output-dir", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_interruption_retains_completed_calibration_checks(
    setup: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(*args: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(checker, "_cameras", interrupt)
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 2 and not report.complete
    assert report.coverage["calibration_files"] == 2
    assert len([f for f in report.findings if f.check_id == "setup.calibration.contract"]) == 2


@pytest.mark.parametrize("content", [b"not json", b'{"id": 1, "id": 2}', b"\xff"])
def test_unreadable_calibration_contents_are_faults(setup: dict[str, Any], content: bytes) -> None:
    path = Path(setup["robot"]["calibration_dir"]) / "arm.json"
    path.write_bytes(content)
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 1 and faults(report, "setup.calibration.file")


def test_calibration_permission_error_is_incomplete(
    setup: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = checker.read_setup_json
    blocked = Path(setup["robot"]["calibration_dir"]) / "arm.json"

    def denied(path: Path) -> Any:
        if path == blocked:
            raise PermissionError("synthetic file read denied")
        return original(path)

    monkeypatch.setattr(checker, "read_setup_json", denied)
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 2 and report.coverage["calibration_files"] == 1


def test_valid_calibration_endpoints(setup: dict[str, Any]) -> None:
    data = calibration()
    data["shoulder_pan"].update(homing_offset=-2047, drive_mode=1)
    data["wrist_roll"].update(homing_offset=2047)
    (Path(setup["robot"]["calibration_dir"]) / "arm.json").write_text(json.dumps(data))
    assert checker.check_so101_setup(setup).exit_code == 0


def test_json_device_paths_are_never_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.open

    def forbidden(path: Path, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Non-regular path was opened")

    path = tmp_path / "directory"
    path.mkdir()
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="regular JSON file"):
        checker.read_setup_json(path)
    monkeypatch.setattr(Path, "open", original)


def test_markdown_does_not_interpret_configuration_markup(setup: dict[str, Any], tmp_path: Path) -> None:
    camera = setup["robot"]["cameras"].pop("front")
    setup["robot"]["cameras"]["<img src=x>`frame`"] = camera
    report = checker.check_so101_setup(setup)
    output = tmp_path / "report"
    checker.write_setup_report(report, output)
    text = (output / "setup.md").read_text()
    assert "<img" not in text and "`frame`" not in text
    assert "&lt;img" in text and "&#96;frame&#96;" in text


def test_report_write_failure_returns_two(
    setup: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(setup))
    original = Path.open

    def fail_markdown(candidate: Path, *args: Any, **kwargs: Any) -> Any:
        if candidate.name == "setup.md":
            raise OSError("synthetic disk error")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_markdown)
    assert main(["--config", str(path), "--output-dir", str(tmp_path / "report")]) == 2


def test_missing_git_leaves_revision_unknown(setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checker.shutil, "which", lambda name: None)
    report = checker.check_so101_setup(setup)
    assert report.exit_code == 0
    assert report.provenance["checkout_revision"] is None


def test_actual_so101_config_serialization_preserves_native_aliases(setup: dict[str, Any]) -> None:
    import draccus

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.teleoperators.config import TeleoperatorConfig
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    follower = draccus.decode(RobotConfig, setup["robot"])
    leader = draccus.decode(TeleoperatorConfig, setup["teleop"])
    assert isinstance(follower, SO101FollowerConfig)
    assert isinstance(follower.cameras["front"], OpenCVCameraConfig)
    assert isinstance(leader, SO101LeaderConfig)
    saved = json.loads(
        json.dumps(
            {
                "robot": draccus.encode(follower, RobotConfig),
                "teleop": draccus.encode(leader, TeleoperatorConfig),
            }
        )
    )
    report = checker.check_so101_setup(saved)
    assert report.exit_code == 0, report.to_dict()
    assert report.coverage["calibration_files"] == 2
