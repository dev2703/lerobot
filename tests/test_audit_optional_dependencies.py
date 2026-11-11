# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import json
import subprocess
import sys
from pathlib import Path


def test_audit_reports_missing_extras_without_loading_dataset_stack(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 1,
                "total_frames": 3,
                "fps": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "features": {"observation.images.camera": {"dtype": "video"}},
            }
        )
    )
    script = """
import json, sys
sys.modules['datasets'] = None
sys.modules['av'] = None
sys.modules['pyarrow'] = None
from lerobot.datasets.audit import audit_dataset
report = audit_dataset(sys.argv[1])
assert report.exit_code == 2
assert {f.check_id for f in report.findings} >= {'tables.dependencies', 'video.dependencies'}
assert 'lerobot.datasets.lerobot_dataset' not in sys.modules
print(json.dumps(report.to_dict()))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)], capture_output=True, text=True, check=True
    )
    assert not json.loads(result.stdout)["complete"]
