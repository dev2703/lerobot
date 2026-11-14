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

"""Run an offline SO-101 setup inspection: lerobot-check-setup --config run.json --output-dir report."""

import argparse
from pathlib import Path

from lerobot.utils.setup_check import SetupReport, check_so101_setup, read_setup_json, write_setup_report


def main(argv: list[str] | None = None) -> int:
    """Inspect setup files and return 0 (no faults), 1 (faults), or 2 (incomplete)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="Native JSON run configuration; must contain robot"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New directory for setup.json and setup.md"
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("--output-dir must be a new directory; existing paths are never overwritten")
    if args.config.resolve().is_relative_to(args.output_dir.resolve()):
        parser.error("--output-dir must not contain the input configuration path")
    try:
        config = read_setup_json(args.config)
        report = check_so101_setup(config)
    except (OSError, ValueError, UnicodeError) as exc:
        report = SetupReport(complete=False)
        report.add(
            "setup.config.read",
            "configuration",
            "not_checked",
            str(args.config),
            "A readable regular JSON configuration file",
            str(exc),
            "Resolve the input error and rerun inspection.",
        )
    except KeyboardInterrupt:
        report = SetupReport(complete=False)
        report.add(
            "setup.interrupted",
            "configuration",
            "not_checked",
            str(args.config),
            "Completed file inspection",
            "Interrupted",
            "Rerun to inspect all configured files.",
        )
    report.provenance["config_path"] = str(args.config.absolute())
    try:
        write_setup_report(report, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Cannot write setup report: {exc}")
        return 2
    print(
        f"File inspection complete: {report.complete}; exit code {report.exit_code}. Reports: {args.output_dir}"
    )
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
