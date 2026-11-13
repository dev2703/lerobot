# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Audit an existing local LeRobot v3.0 dataset without network access."""

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.audit import audit_dataset


def main(argv: list[str] | None = None) -> int:
    """Write JSON and Markdown evidence and return the audit's exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Existing local dataset directory")
    source.add_argument("--repo-id", help="Resolve an existing cached dataset; never downloads")
    parser.add_argument("--revision", help="Existing cached Hub ref or snapshot")
    parser.add_argument("--output-dir", type=Path, required=True, help="Report directory outside the dataset")
    parser.add_argument(
        "--video-scan",
        choices=("full", "sample", "spot"),
        default="full",
        help="full/sample audit all rows and decode all video; spot skips row scans and probes bounded video regions",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--timestamp-tolerance-s", type=float)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.root is not None and args.output_dir.expanduser().resolve().is_relative_to(
        args.root.expanduser().resolve()
    ):
        parser.error("--output-dir must be outside the dataset root")
    report = audit_dataset(
        args.root,
        repo_id=args.repo_id,
        revision=args.revision,
        video_scan=args.video_scan,
        sample_seed=args.sample_seed,
        timestamp_tolerance_s=args.timestamp_tolerance_s,
    )
    try:
        report.write(args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Cannot write audit reports: {exc}", file=sys.stderr)
        return 2
    print(
        f"Audit {'complete' if report.complete else 'partial'}; exit={report.exit_code}; reports: {args.output_dir}"
    )
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
