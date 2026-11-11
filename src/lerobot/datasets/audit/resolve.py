# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Resolve existing local cache entries without importing Hub clients."""

import os
from pathlib import Path


class InputResolutionError(ValueError):
    """An offline input cannot be uniquely resolved."""

    def __init__(self, message: str, candidates: list[str] | None = None):
        super().__init__(message)
        self.candidates = candidates or []


def resolve_input(
    root: str | Path | None,
    repo_id: str | None,
    revision: str | None,
) -> tuple[Path, dict[str, object]]:
    """Resolve a directory, a legacy cache, or a revision-safe Hub snapshot."""
    if (root is None) == (repo_id is None):
        raise InputResolutionError("Provide exactly one of root or repo_id.")
    if root is not None:
        if revision is not None:
            raise InputResolutionError("revision applies only to repo_id cache resolution.")
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise InputResolutionError(f"Local dataset directory does not exist: {path}")
        return path, {"kind": "local", "root": str(path)}
    assert repo_id is not None
    parts = repo_id.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") or "\\" in part for part in parts):
        raise InputResolutionError("repo_id must have the form owner/dataset.")
    if revision is not None and (Path(revision).is_absolute() or ".." in Path(revision).parts):
        raise InputResolutionError("revision must be a cache ref or snapshot name.")
    cache_home = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    hf_home = Path(os.getenv("HF_HOME", str(cache_home / "huggingface"))).expanduser()
    lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", str(hf_home / "lerobot"))).expanduser()
    hub_home = Path(
        os.getenv("HF_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))
    ).expanduser()
    candidates: set[Path] = set()
    legacy = lerobot_home / repo_id
    if revision is None and legacy.is_dir():
        candidates.add(legacy.resolve())
    for hub_cache in {hub_home, lerobot_home / "hub"}:
        repository = hub_cache / ("datasets--" + repo_id.replace("/", "--"))
        snapshots = repository / "snapshots"
        if revision is None:
            if snapshots.is_dir():
                candidates.update(path.resolve() for path in snapshots.iterdir() if path.is_dir())
            continue
        snapshot_name = revision
        ref = repository / "refs" / revision
        if ref.is_file():
            try:
                snapshot_name = ref.read_text().strip()
            except OSError as exc:
                raise InputResolutionError(f"Cannot read cached revision ref: {exc}") from exc
        if not snapshot_name or "/" in snapshot_name or "\\" in snapshot_name or snapshot_name in (".", ".."):
            raise InputResolutionError("Cached revision ref is invalid.")
        snapshot = snapshots / snapshot_name
        if snapshot.is_dir():
            candidates.add(snapshot.resolve())
    paths = sorted(str(path) for path in candidates)
    if len(paths) != 1:
        reason = (
            "No existing cached dataset found"
            if not paths
            else "Several cached candidates exist; select an explicit root or revision"
        )
        raise InputResolutionError(
            reason + ". No network lookup was attempted; cache omissions say nothing about remote integrity.",
            paths,
        )
    path = Path(paths[0])
    return path, {"kind": "cache", "repo_id": repo_id, "revision": revision, "root": str(path)}
